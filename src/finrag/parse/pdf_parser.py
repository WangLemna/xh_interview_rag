"""解析主流程：把页级提取结果串成结构 IR；只产出 IR，不做跨页合并、不做 OCR。
逐页 scan + 准入 → 跨页重复统计与噪声剔除 → 栏结构分析（默认不重排）→ 标题层级（TOC 主路径 / 版式兜底）→ 组装块（表格结构化、图片占位块按 y0 插入）→ ``ParsedDoc``（+ ``validate()`` 自检）。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Sequence

import fitz  # PyMuPDF

from finrag.config import AppConfig, ParseConfig, get_config
from finrag.parse import heading as H
from finrag.parse import layout as L
from finrag.parse import table as T
from finrag.parse.guard import (
    MULTICOL_WIDTH_RATIO,
    PROBE_VERSION,
    REPEAT_MIN_RATIO,
    ZONE_BODY,
    ZONE_FOOTER,
    ZONE_HEADER,
    Admission,
    can_early_reject,
    edge_zone,
    is_empty_page,
    iter_repeat_keys,
    judge_admission,
    norm_block_text,
)
from finrag.parse.ir import (
    IR_SCHEMA,
    PARSER_VERSION,
    BBox,
    Block,
    DocMeta,
    PageRef,
    ParsedDoc,
    TableData,
    sanitize_text,
)
from finrag.parse.pageio import PageScan, printed_page_label, scan_page

__all__ = ["ParseResult", "parse_pdf"]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ParseResult:
    """一次解析的完整结果。被准入拒绝时 ``doc`` 为 None，其余字段仍填好（拒绝原因本身也是产出）。"""

    path: Path
    admission: Admission
    doc: ParsedDoc | None = None
    pages_scanned: int = 0
    # IR 里实际落下的块数（不含被拒文档）。结构化表格把 N 个块折叠成 1 个，故它会
    # 小于"保留块数"：差值 = Σ(每张成功结构化的表吸收的块数 − 1)，那是折叠，不是丢内容。
    blocks_kept: int = 0
    blocks_dropped: int = 0
    headings: int = 0
    noise_ratio: float = 0.0  # 被剔除的块占全部块的比例
    warnings: list[str] = field(default_factory=list)
    #: 解析在框架层失败（坏 PDF / 加密 / 内存不足）时的原因；非空则 ``doc`` 恒为 None。
    #: 与"被准入拒绝"不同 —— 后者是判据给出的结论，前者是根本没跑起来。
    fatal: str = ""

    @property
    def ok(self) -> bool:
        return self.doc is not None


# ------------------------------------------------------------------
# 噪声剔除
# ------------------------------------------------------------------
def _repeat_counts(scans: Sequence[PageScan]) -> tuple[dict[str, int], int]:
    """跨页重复统计。分母是**非空页数**（空页不计入，与语料体检口径一致）。"""
    text_pages = [sc for sc in scans if not is_empty_page(sc.n_chars)]
    counts: dict[str, int] = {}
    for sc in text_pages:
        for key, _b in iter_repeat_keys(sc.blocks):
            counts[key] = counts.get(key, 0) + 1
    return counts, len(text_pages)


def _is_noise(
    block: Any, page_height: float, counts: dict[str, int], n_text_pages: int, pcfg: ParseConfig
) -> bool:
    """这一块该不该整块丢掉：页边（页眉 / 页脚）只看跨页重复率是否达标。
    正文区更严（跨页重复率 ≥ ``repeat_strong_ratio`` 且归一化长度 ≤ ``repeat_strong_maxlen``），宁可留噪声也不误删表头。"""
    if n_text_pages <= 0 or page_height <= 0:
        return False
    key = norm_block_text(block.text)
    ratio = counts.get(key, 0) / n_text_pages
    zone = edge_zone(block.bbox.y0, page_height)
    if zone in (ZONE_HEADER, ZONE_FOOTER):
        return ratio >= REPEAT_MIN_RATIO
    if zone == ZONE_BODY:
        return len(key) <= pcfg.repeat_strong_maxlen and ratio >= pcfg.repeat_strong_ratio
    return False


def _split_blocks(
    scans: Sequence[PageScan], counts: dict[str, int], n_text_pages: int, pcfg: ParseConfig
) -> tuple[list[list[Any]], int, int]:
    """按页产出**保留下来的正文文本块**。返回 (每页块表, 保留数, 剔除数)。"""
    kept: list[list[Any]] = []
    n_kept = 0
    n_dropped = 0
    for sc in scans:
        page_blocks = []
        for b in sc.blocks:
            if not b.text.strip():
                continue
            if _is_noise(b, sc.height, counts, n_text_pages, pcfg):
                n_dropped += 1
                continue
            page_blocks.append(b)
            n_kept += 1
        kept.append(page_blocks)
    return kept, n_kept, n_dropped


# ------------------------------------------------------------------
# 栏重排（默认关闭，见 layout.py）
# ------------------------------------------------------------------
def _column_order(
    raw: fitz.Document, scan: PageScan, kept: list[Any]
) -> list[int] | None:
    """这一页要不要重排；要则给出完整块表的新阅读顺序，否则 None。
    只对 ``edge_zone == ZONE_BODY`` 的块分析（须与探针一致）；非正文区残留直接排到最后，且内部会自行再调一次 ``find_tables``。"""
    if scan.height <= 0:
        return None
    body_idx = [
        i for i, b in enumerate(kept) if edge_zone(b.bbox.y0, scan.height) == ZONE_BODY
    ]
    body = [kept[i] for i in body_idx]
    pc = L.analyze(body)
    if pc is None:
        return None
    try:
        tabs = raw[scan.page_no - 1].find_tables()
        boxes = [tuple(float(v) for v in t.bbox) for t in tabs.tables]
    except Exception as exc:  # find_tables 在个别页上会抛
        logger.debug("find_tables 失败 p%d: %s", scan.page_no, exc)
        boxes = []
    centers = [(gs + ge) / 2 for gs, ge in pc.gutters]
    covered = any(bx[0] <= c <= bx[2] for bx in boxes for c in centers)
    if not L.should_reorder(pc, covered):
        return None
    new_body = [body_idx[k] for k in L.column_major_order(body, pc.cols)]
    body_set = set(body_idx)
    tail = [i for i in range(len(kept)) if i not in body_set]
    return new_body + tail


# ------------------------------------------------------------------
# 表格：检出（find_tables）+ 结构化（extract）+ 守门
# ------------------------------------------------------------------
class _PageTable(NamedTuple):
    """本页的一张 ``find_tables`` 表：矩形 + 结构化产出，两种状态由 ``data`` 决定。
    ``data`` 为 None ⇒ 保留原块文本、kind 标 table；非 None ⇒ 守门通过，N 个块折叠成一个表格块。"""

    bbox: BBox
    data: TableData | None


def _center_in(bbox: BBox, box: BBox) -> bool:
    """按中心点判断块是否落在表格矩形内 —— ``_is_table_block`` 的同一判据。窄框表上不成立，是低 align_recall 的成因，但一个字符都没丢。"""
    cx = (bbox.x0 + bbox.x1) / 2
    cy = (bbox.y0 + bbox.y1) / 2
    return box.x0 <= cx <= box.x1 and box.y0 <= cy <= box.y1


def _dim(value: Any, fallback: int) -> int:
    """取 ``row_count`` / ``col_count``，取不到或非正时退回矩阵实测值。"""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return fallback
    return v if v > 0 else fallback


def _header_info(t: Any) -> tuple[list[str], bool]:
    """``t.header`` → ``(扁平化的列名, 表头是否在表格矩形之外)``。读不到就返回 ``([], False)``，表头取不到不影响结构化成立。"""
    try:
        hdr = t.header
    except Exception:  # noqa: BLE001 —— 个别表上取 header 会抛
        return [], False
    # 表头名同样直出 ``t.header`` ⇒ 与 extract 单元格同一道清洗。
    names = [sanitize_text(n) for n in T.flatten_header_names(getattr(hdr, "names", None) or [])]
    return names, bool(getattr(hdr, "external", False))


def _read_table(t: Any, block_texts: Sequence[str]) -> TableData | None:
    """一张表 → 通过守门的 :class:`TableData`；任一环节不过关都返回 None（不抛，属"这一张没拿到"）。
    顺序不能换：先对账（拿 ``extract()`` 的原始输出与块文本比）后裁空。"""
    try:
        raw = t.extract()
    except Exception as exc:  # noqa: BLE001
        logger.debug("extract 失败：%s: %s", type(exc).__name__, exc)
        return None
    if not raw:
        return None
    # 单元格文本直出 ``extract()``、没过 pageio ⇒ 在此独立清洗（NUL 会被 PG 拒收）。
    data = [["" if c is None else sanitize_text(str(c)) for c in (row or [])] for row in raw]

    rec = T.reconcile(data, block_texts)
    if not rec.ok:
        # 守门。这是"不丢内容"的全部保证 —— 一旦被绕过，丢的内容不会报错，
        # 只会静静地不在语料里。
        return None

    rows, keep = T.drop_empty(data)
    if not rows:
        # 整张表都是空单元格：extract 拿到了形状却没拿到内容，不能作为结构化产出。
        return None

    header, external = _header_info(t)
    # 表头从 extract 的列坐标系投影到裁空后的坐标系。投影失败（长度对不上）时
    # align_header 返回空表 —— 宁可没有列名，也不能给一份错位的。
    header = T.align_header(header, keep, len(rows[0]))
    n_cols_fallback = max(len(r) for r in data)
    return TableData(
        rows=rows,
        n_rows_raw=_dim(getattr(t, "row_count", None), len(data)),
        n_cols_raw=_dim(getattr(t, "col_count", None), n_cols_fallback),
        header=header,
        header_external=external,
        align_recall=rec.align_recall,
    )


def _page_tables(
    raw: fitz.Document,
    page_no: int,
    page_blocks: Sequence[Any],
    structure: bool,
) -> tuple[list[_PageTable], int]:
    """本页 ``find_tables`` 的表 + 结构化结果，以及**标注盲区**张数；只在这一层调 ``find_tables``。
    盲区（检出表但无块中心落在矩形内）刻意不结构化、只计数；``structure=False`` 时仍照跑检出、只停结构化；失败不抛。"""
    try:
        tabs = list(raw[page_no - 1].find_tables().tables)
    except Exception as exc:  # noqa: BLE001 —— find_tables 在个别页上会抛
        logger.debug("find_tables 失败 p%d: %s", page_no, exc)
        return [], 0

    out: list[_PageTable] = []
    n_orphan = 0
    for t in tabs:
        try:
            box = BBox(*(float(v) for v in t.bbox))
        except Exception:  # noqa: BLE001 —— bbox 取不到的表跳过，不影响其余
            continue
        # 这里的下标**只用于**取对账用的块文本，不往外传（理由见 _PageTable）。
        # 重排是置换 ⇒ 块文本的多重集不变 ⇒ 与调用时机无关。
        inside = [i for i, b in enumerate(page_blocks) if _center_in(b.bbox, box)]
        data = None
        if not inside:
            n_orphan += 1
        elif structure:
            data = _read_table(t, [page_blocks[i].text for i in inside])
        out.append(_PageTable(bbox=box, data=data))
    return out, n_orphan


def _is_table_block(bbox: BBox, table_boxes: Sequence[BBox]) -> bool:
    """文本块的中心点是否落在某张表格矩形内 —— 是则标 ``kind="table"``（判据与 ``_center_in`` 逐字相同）。
    ``kind="table"`` 的准确读法是"这个块落在某张表格矩形内"，而非"这个块是一张表"（一张表常罩住多个块）。"""
    if not table_boxes:
        return False
    # 判据本体在 ``_center_in`` —— 这里只是"任一张表命中即可"的折叠。
    # 别在本函数里重写中心点计算：两处判据必须逐字相同。
    return any(_center_in(bbox, t) for t in table_boxes)


# ------------------------------------------------------------------
# 事件流：把标题与块按阅读顺序排成一条
# ------------------------------------------------------------------
def _page_items(blocks: list[Block], hits: list[H.HeadingHit]) -> list[Any]:
    """把一页的（块, 标题）按阅读顺序交成一条流：用游标推进，不按 y0 全量重排（会抹掉已定下的块顺序）。
    标题插在第一个 y0 不小于它的块之前；TOC 路径 y0 为 -1，自然落在页首。"""
    items: list[Any] = []
    cursor = 0
    for hit in sorted(hits, key=lambda h: h.y0):
        while cursor < len(blocks) and blocks[cursor].bbox.y0 < hit.y0:
            items.append(blocks[cursor])
            cursor += 1
        items.append(hit)
    items.extend(blocks[cursor:])
    return items


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def parse_pdf(pdf_path: str | Path, cfg: AppConfig | None = None) -> ParseResult:
    cfg = cfg or get_config()
    pcfg = cfg.parse
    path = Path(pdf_path)
    warnings: list[str] = []

    scans: list[PageScan] = []
    empty_pages = 0
    toc_raw: list[Any] = []

    with fitz.open(path) as raw:
        page_count = raw.page_count
        try:
            toc_raw = list(raw.get_toc(simple=True) or [])
        except Exception as exc:
            warnings.append(f"书签读取失败（{type(exc).__name__}）：改用版式兜底")
            toc_raw = []

        # ---- 1. 逐页扫描 + 准入早退 ----
        for i in range(page_count):
            sc = scan_page(raw[i], i + 1)
            scans.append(sc)
            if is_empty_page(sc.n_chars):
                empty_pages += 1
            if can_early_reject(empty_pages, page_count):
                warnings.append(
                    f"提前判定为纯扫描件：已扫 {i + 1}/{page_count} 页时空页数下界已越过 "
                    f"SCAN_PURE_RATIO，故空页数 {empty_pages} 是全量下的下界（实际可能更高）"
                )
                break

        # ---- 2. 准入判定。分母用**文档总页数**，早退时 empty_pages 只是下界 ----
        admission = judge_admission(empty_pages, page_count)
        if not admission.eligible:
            return ParseResult(
                path=path,
                admission=admission,
                pages_scanned=len(scans),
                warnings=warnings,
            )

        # ---- 3. 噪声剔除 ----
        counts, n_text_pages = _repeat_counts(scans)
        kept, n_kept, n_dropped = _split_blocks(scans, counts, n_text_pages, pcfg)

        # ---- 4. 栏重排（默认关）----
        orders: dict[int, list[int]] = {}
        if pcfg.reorder_columns:
            for sc, page_blocks in zip(scans, kept):
                order = _column_order(raw, sc, page_blocks)
                if order is not None:
                    orders[sc.page_no] = order
            if orders:
                warnings.append(
                    f"栏重排启用：{len(orders)} 页被重新排序（默认关闭，见 layout.py 实测结论）"
                )

        # ---- 5. 标题层级：TOC 主路径 / 版式兜底，收束到同一棵树 ----
        body_size, body_share = H.body_size_stats(kept)
        # warnings 兼作 notes 出口：书签页码被钳制时必须留痕，
        # 否则"修正"就只是换了个方式静默改数据。
        toc_hits = H.toc_hits(toc_raw, page_count, warnings)
        used_toc = bool(toc_hits)
        if used_toc:
            hits = toc_hits
            source = "toc"
        elif pcfg.fallback_enabled:
            hits = H.layout_hits(kept)
            source = "layout"
            if body_share < pcfg.min_body_share:
                warnings.append(
                    f"无 TOC，走版式兜底且低置信：body_share={body_share:.2f} < "
                    f"{pcfg.min_body_share}（主字号漂移严重，字号差推不出标题；"
                    f"实测同类文档精确率仅 42%）"
                )
        else:
            hits = []
            source = "layout"
            warnings.append("无 TOC 且版式兜底已禁用（parse.fallback_enabled=false）：全文归入虚拟根")

        # 逐页组装块 + 事件流
        hits_by_page: dict[int, list[H.HeadingHit]] = {}
        for h in hits:
            hits_by_page.setdefault(h.page, []).append(h)
        builder = H.TreeBuilder(source=source)  # type: ignore[arg-type]
        blocks_kept_ir = 0
        n_tables_seen = 0     # find_tables 检出的表（含盲区）
        n_tables_struct = 0   # 其中守门通过、被折叠的
        n_tables_orphan = 0   # 其中没有任何块落在矩形内的（标注盲区）
        for sc, page_blocks in zip(scans, kept):
            order = orders.get(sc.page_no)
            if order is not None:
                page_blocks = [page_blocks[i] for i in order]
            # 表格框只对**有保留块**的页去检：空页不可能有表格，而 find_tables
            # 比 get_text 慢一个数量级。
            page_tables, n_orphan = (
                _page_tables(raw, sc.page_no, page_blocks, pcfg.structure_tables)
                if page_blocks
                else ([], 0)
            )
            n_tables_seen += len(page_tables)
            n_tables_orphan += n_orphan
            n_tables_struct += sum(1 for t in page_tables if t.data is not None)
            ir_blocks = _to_ir_blocks(sc, page_blocks, page_tables)
            for item in _page_items(ir_blocks, hits_by_page.get(sc.page_no, [])):
                if isinstance(item, H.HeadingHit):
                    builder.push(item)
                else:
                    builder.current.blocks.append(item)
                    blocks_kept_ir += 1

        # ---- 5.1 表格产出的留痕 ----
        # 三个数各自对应一个不同的失效模式，混在一起就看不出来：
        #   结构化成功 —— 真正生效的部分
        #   守门未过   —— 结构化被拒、退回未结构化形态（内容没丢，只是没结构化）
        #   标注盲区   —— 表检出了但内容根本不在 IR 里（这块内容一直是缺的）
        if pcfg.structure_tables:
            warnings.append(
                f"表格结构化：检出 {n_tables_seen} 张，成功 {n_tables_struct} 张，"
                f"守门未过 {n_tables_seen - n_tables_struct - n_tables_orphan} 张"
                f"（退回未结构化形态，内容未丢）"
            )
        elif n_tables_seen:
            warnings.append(
                f"表格结构化已关闭（parse.structure_tables=false）："
                f"检出 {n_tables_seen} 张，全部保留 D10 的未结构化形态"
            )
        if n_tables_orphan:
            warnings.append(
                f"表格标注盲区：{n_tables_orphan} 张表没有任何块落在矩形内，"
                f"其内容未进 IR —— 它此前是完全不可见的量，本次仅计数、不作处置"
            )

        # ---- 6. 元数据与页表 ----
        # multicol_ratio 用的仍是"页级代理"（median_rel_width 阈值），不是 layout 检出的
        # 栏数；保这个口径是为了与语料体检的字段对得上。它的"多栏"大多实为表格单元格，
        # 别当栏数用。
        multicol = sum(
            1
            for sc in scans
            if sc.height > 0 and sc.n_blocks >= 4 and sc.median_rel_width < MULTICOL_WIDTH_RATIO
        )
        meta = DocMeta(
            ir_schema=IR_SCHEMA,
            parser_version=PARSER_VERSION,
            probe_version=PROBE_VERSION,
            doc_type=admission.doc_type,
            # 用 used_toc 而不是"书签条数 > 0"：本字段标记**实际走了哪条路径**。
            # 书签全被过滤掉时实际走的是兜底路径，此时记 True 会误导下游。
            has_toc=used_toc,
            page_count=page_count,
            multicol_ratio=round(multicol / len(scans), 4) if scans else 0.0,
            body_share=round(body_share, 4),
        )
        pages = [PageRef(index=i + 1, label=printed_page_label(sc)) for i, sc in enumerate(scans)]

    doc = ParsedDoc(source_path=path, root=builder.root, pages=pages, meta=meta)

    # ---- 7. 结构自检。不抛异常：要的是"哪几条不对"的清单 ----
    problems = doc.validate()
    if problems:
        warnings.extend(f"IR 自检未通过：{p}" for p in problems[:10])
        logger.warning("%s IR 自检发现 %d 处问题", path.name, len(problems))

    total_blocks = n_kept + n_dropped
    return ParseResult(
        path=path,
        admission=admission,
        doc=doc,
        pages_scanned=len(scans),
        blocks_kept=blocks_kept_ir,
        blocks_dropped=n_dropped,
        headings=len(hits),
        noise_ratio=round(n_dropped / total_blocks, 4) if total_blocks else 0.0,
        warnings=warnings,
    )


def _to_ir_blocks(
    scan: PageScan, page_blocks: list[Any], page_tables: Sequence[_PageTable] = ()
) -> list[Block]:
    """页内文本块 + 表格块 + 图片占位块 → IR Block 序列（阅读顺序）；图片只按 y0 插入，不改文本块相对顺序。
    表格形态由 ``_PageTable.data``（不是 ``kind``）决定：非空则把罩住的 N 个块折叠成一个表格块（会减少块总数，属预期），为空则只把中心点在未结构化表格矩形内的块标成 ``kind="table"``。"""
    # 块下标 → 认领它的表。**按 bbox 重算，不接收调用方传来的下标**：块顺序可能已被
    # 栏重排动过，下标错位轻则折叠的块不是这张表、重则内容从 IR 里消失，且不报错。
    #
    # **先到先得**：表格矩形互相重叠时，不让后一张把块抢走。
    owner: dict[int, int] = {}
    for ti, t in enumerate(page_tables):
        if t.data is None:
            continue
        for bi, b in enumerate(page_blocks):
            if bi not in owner and _center_in(b.bbox, t.bbox):
                owner[bi] = ti
    # 未结构化的表格矩形 —— 给"没被吸收的块"标 kind。
    fallen = [t.bbox for t in page_tables if t.data is None]

    text_blocks: list[Block] = []
    absorbed: set[int] = set()
    for bi, b in enumerate(page_blocks):
        ti = owner.get(bi)
        if ti is None:
            text_blocks.append(
                Block(
                    text=b.text,
                    page=scan.page_no,
                    bbox=b.bbox,
                    kind="table" if _is_table_block(b.bbox, fallen) else "paragraph",
                    font_size=b.size,
                    bold=b.bold,
                    n_lines=b.n_lines,
                    # 行几何透传（``IR_SCHEMA`` 4），只透传不解释。结构化表格块（下面
                    # 那条分支）刻意不填：它的 text 是 markdown，本就有换行与管道符。
                    lines=b.lines,
                )
            )
            continue
        if ti in absorbed:
            continue  # 已被这张表吸收过，本轮只输出一个表格块
        absorbed.add(ti)
        t = page_tables[ti]
        # 走到这里必然有数据：owner 只登记 data 非空的表。
        if t.data is None:  # pragma: no cover —— 枚举上的不可达，留作断言
            continue
        # 表格块落在**本张表第一个被认领的块**的位置上，于是它在阅读顺序里的
        # 落点与折叠前一致。
        head = b
        text_blocks.append(
            Block(
                text=T.serialize_md(t.data.rows),
                page=scan.page_no,
                bbox=t.bbox,  # 表格矩形，不是首行的
                kind="table",
                # 字号 / 加粗取首个被吸收块的值：表格块在这两个字段上没有单一真值，
                # 取表首行最接近"这块看起来多大"。它们不参与任何表格判据。
                font_size=head.size,
                bold=head.bold,
                # n_lines 取真值 = 表格记录数，别沿用首块的 "1"。
                n_lines=len(t.data.rows),
                table=t.data,
            )
        )

    if not scan.image_boxes:
        return text_blocks

    out: list[Block] = []
    cursor = 0
    for ibox in sorted(scan.image_boxes, key=lambda x: x.y0):
        while cursor < len(text_blocks) and text_blocks[cursor].bbox.y0 < ibox.y0:
            out.append(text_blocks[cursor])
            cursor += 1
        out.append(Block(text="", page=scan.page_no, bbox=ibox, kind="image"))
    out.extend(text_blocks[cursor:])
    return out
