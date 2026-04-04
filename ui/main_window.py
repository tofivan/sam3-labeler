"""
MainWindow v2 — Full features + optimized UI

Features: output format settings, slider controls, background inference, class delete, number key shortcuts, progress display
"""
import cv2
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QComboBox, QRadioButton,
    QButtonGroup, QCheckBox, QSlider, QListWidget, QListWidgetItem,
    QGroupBox, QFileDialog, QSplitter, QScrollArea, QFrame,
    QAbstractItemView, QProgressBar, QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QShortcut, QKeySequence, QIcon, QColor, QPixmap, QPainter

from ui.canvas import AnnotationCanvas
from core.state import LabelingState
from core.io_manager import (
    load_config, save_config, save_progress, load_progress,
    persist_classes, load_persisted_classes, load_existing_labels,
    auto_save_labels,
)
from core.sam_engine import SAMEngine


# -- helpers -----------------------------------------------------------

def color_icon(color: QColor, size=12):
    pm = QPixmap(size, size); pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm); p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(color); p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(0, 0, size, size, 3, 3); p.end()
    return QIcon(pm)


def _section_label(text):
    lbl = QLabel(text)
    lbl.setStyleSheet("color:#8892b0; font-size:11px; font-weight:600; padding:6px 0 2px 0;")
    return lbl


from ui.canvas import LABEL_COLORS


# -- SAM Worker --------------------------------------------------------

class SAMWorker(QThread):
    finished = pyqtSignal(object, str)

    def __init__(self, func, *args):
        super().__init__()
        self._func = func; self._args = args

    def run(self):
        try:
            result = self._func(*self._args)
            self.finished.emit(result, "")
        except Exception as e:
            self.finished.emit(None, str(e))


# -- MainWindow --------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, sam_model_path="sam3.pt"):
        super().__init__()
        self.setWindowTitle("SAM3 Labeler")
        self.setMinimumSize(1400, 900)
        self.state = LabelingState()
        self.state.classes = load_persisted_classes()
        self.sam = SAMEngine(model_path=sam_model_path)
        self._worker = None
        config = load_config()

        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(4,4,4,4); root.setSpacing(4)

        # ======= Navigation bar =======
        nav = QFrame(); nav.setObjectName("navBar")
        nav_lay = QHBoxLayout(nav); nav_lay.setContentsMargins(8,6,8,6); nav_lay.setSpacing(6)

        logo = QLabel("SAM3"); logo.setStyleSheet("font-size:15px; font-weight:800; color:#7c8cf5;")
        nav_lay.addWidget(logo)
        self.folder_input = QLineEdit(config.get("images_folder",""))
        self.folder_input.setPlaceholderText("Image folder"); nav_lay.addWidget(self.folder_input, 4)
        browse_btn = QPushButton("📁"); browse_btn.setFixedWidth(36)
        browse_btn.setToolTip("Browse folder"); browse_btn.clicked.connect(self._browse_folder)
        nav_lay.addWidget(browse_btn)
        self.output_input = QLineEdit(config.get("output_folder",""))
        self.output_input.setPlaceholderText("Output folder"); nav_lay.addWidget(self.output_input, 3)
        load_btn = QPushButton("Load"); load_btn.setObjectName("primaryBtn")
        load_btn.clicked.connect(self._load_folder); nav_lay.addWidget(load_btn)

        nav_lay.addWidget(QLabel("│"))
        self.jump_input = QLineEdit(); self.jump_input.setPlaceholderText("N")
        self.jump_input.setFixedWidth(48); self.jump_input.returnPressed.connect(self._jump_to)
        nav_lay.addWidget(self.jump_input)
        jump_btn = QPushButton("Go"); jump_btn.clicked.connect(self._jump_to)
        nav_lay.addWidget(jump_btn)
        self.nav_label = QLabel("0 / 0"); self.nav_label.setStyleSheet("color:#8892b0; font-size:12px;")
        nav_lay.addWidget(self.nav_label)
        root.addWidget(nav)

        # ======= Body (tools | canvas | sidebar) =======
        body = QSplitter(Qt.Orientation.Horizontal)

        # -- Tool panel --
        tool_w = QFrame(); tool_w.setObjectName("toolPanel"); tool_w.setFixedWidth(120)
        tl = QVBoxLayout(tool_w); tl.setContentsMargins(6,12,6,12); tl.setSpacing(6)
        self.mode_group = QButtonGroup(self)
        for i, (label, mode, tip) in enumerate([
            ("🖱️ Click", "click", "Click segmentation (1)"),
            ("⬜ Box", "box", "Box selection (2)"),
            ("✋ Select", "select", "Select annotations (3)"),
        ]):
            rb = QRadioButton(label); rb.setProperty("mode", mode); rb.setToolTip(tip)
            rb.setStyleSheet("font-size: 16px; padding: 6px 2px;")
            if i==0: rb.setChecked(True)
            self.mode_group.addButton(rb, i); tl.addWidget(rb)
        self.mode_group.buttonClicked.connect(self._mode_changed)
        tl.addStretch()

        fit_btn = QPushButton("⊞ Fit"); fit_btn.setToolTip("Fit to window (F)")
        fit_btn.clicked.connect(lambda: self.canvas.fit_view())
        tl.addWidget(fit_btn)
        prev_btn = QPushButton("◀ Prev"); prev_btn.clicked.connect(self._prev_image)
        next_btn = QPushButton("Next ▶"); next_btn.clicked.connect(self._next_image)
        tl.addWidget(prev_btn); tl.addWidget(next_btn)
        body.addWidget(tool_w)

        # -- Canvas --
        self.canvas = AnnotationCanvas()
        self.canvas.point_clicked.connect(self._on_point_click)
        self.canvas.box_drawn.connect(self._on_box_drawn)
        self.canvas.label_selected.connect(self._on_label_selected)
        body.addWidget(self.canvas)

        # -- Sidebar (scrollable) --
        sidebar_scroll = QScrollArea(); sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll.setFixedWidth(340)
        sidebar = QWidget(); sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(6,6,6,6); sl.setSpacing(4)

        # -- Text segmentation --
        sl.addWidget(_section_label("Text Prompt"))
        self.text_prompt = QLineEdit(self.state.classes[0] if self.state.classes else "")
        self.text_prompt.setPlaceholderText("e.g. debris, diver, boat")
        self.text_prompt.returnPressed.connect(self._segment_text)
        sl.addWidget(self.text_prompt)
        seg_btn = QPushButton("▶ Run Text Segmentation"); seg_btn.setObjectName("primaryBtn")
        seg_btn.clicked.connect(self._segment_text); sl.addWidget(seg_btn)

        # -- Class management --
        sl.addWidget(_section_label("Class"))
        self.class_combo = QComboBox(); self.class_combo.addItems(self.state.classes)
        sl.addWidget(self.class_combo)
        add_r = QHBoxLayout()
        self.new_class_input = QLineEdit(); self.new_class_input.setPlaceholderText("New class")
        self.new_class_input.returnPressed.connect(self._add_class)
        add_r.addWidget(self.new_class_input)
        add_btn = QPushButton("+"); add_btn.setFixedWidth(32); add_btn.clicked.connect(self._add_class)
        add_r.addWidget(add_btn)
        del_cls_btn = QPushButton("−"); del_cls_btn.setFixedWidth(32)
        del_cls_btn.setToolTip("Delete current class"); del_cls_btn.clicked.connect(self._delete_class)
        add_r.addWidget(del_cls_btn)
        sl.addLayout(add_r)

        # -- Settings --
        sl.addWidget(_section_label("Settings"))
        self.fallback_cb = QCheckBox("Fallback to box if SAM fails"); self.fallback_cb.setChecked(True)
        sl.addWidget(self.fallback_cb)

        dm_r = QHBoxLayout(); dm_r.setSpacing(2)
        self.dm_group = QButtonGroup(self)
        for label, mode in [("Outline","outline"),("Mask","mask"),("Both","both")]:
            rb = QRadioButton(label); rb.setProperty("dm", mode)
            if mode=="outline": rb.setChecked(True)
            self.dm_group.addButton(rb); dm_r.addWidget(rb)
        self.dm_group.buttonClicked.connect(self._display_mode_changed)
        sl.addLayout(dm_r)

        # Polygon simplification slider
        sl.addWidget(QLabel("Polygon Simplify"))
        self.epsilon_slider = QSlider(Qt.Orientation.Horizontal)
        self.epsilon_slider.setRange(1, 20); self.epsilon_slider.setValue(5)
        self.epsilon_slider.setToolTip("Lower = more precise (0.001~0.020)")
        self.epsilon_label = QLabel("0.005")
        eps_r = QHBoxLayout(); eps_r.addWidget(self.epsilon_slider); eps_r.addWidget(self.epsilon_label)
        self.epsilon_slider.valueChanged.connect(self._epsilon_changed)
        sl.addLayout(eps_r)

        # Overlap threshold slider
        sl.addWidget(QLabel("Overlap Threshold"))
        self.overlap_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlap_slider.setRange(0, 50); self.overlap_slider.setValue(10)
        self.overlap_label = QLabel("10%")
        ov_r = QHBoxLayout(); ov_r.addWidget(self.overlap_slider); ov_r.addWidget(self.overlap_label)
        self.overlap_slider.valueChanged.connect(self._overlap_changed)
        sl.addLayout(ov_r)

        # Output formats
        sl.addWidget(_section_label("Output Formats"))
        self.fmt_obb = QCheckBox("OBB (Oriented Box)"); self.fmt_obb.setChecked(True)
        self.fmt_seg = QCheckBox("YOLO-Seg (Polygon)"); self.fmt_seg.setChecked(True)
        self.fmt_mask = QCheckBox("PNG Mask")
        self.fmt_coco = QCheckBox("COCO JSON")
        for cb in (self.fmt_obb, self.fmt_seg, self.fmt_mask, self.fmt_coco):
            cb.stateChanged.connect(self._fmt_changed); sl.addWidget(cb)

        # -- Annotation list --
        sl.addWidget(_section_label("Annotations"))
        self.label_list = QListWidget()
        self.label_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.label_list.setMaximumHeight(300)
        self.label_list.itemSelectionChanged.connect(self._on_list_selection)
        sl.addWidget(self.label_list)

        # Action row
        op1 = QHBoxLayout(); op1.setSpacing(4)
        self.change_combo = QComboBox(); self.change_combo.addItems(self.state.classes)
        self.change_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        op1.addWidget(self.change_combo)
        chg_btn = QPushButton("Apply Class"); chg_btn.clicked.connect(self._change_selected_class)
        op1.addWidget(chg_btn)
        sl.addLayout(op1)

        op2 = QHBoxLayout(); op2.setSpacing(4)
        del_btn = QPushButton("🗑 Delete"); del_btn.clicked.connect(self._delete_selected)
        clr_btn = QPushButton("Clear All"); clr_btn.clicked.connect(self._clear_all)
        op2.addWidget(del_btn); op2.addWidget(clr_btn)
        sl.addLayout(op2)

        # Save
        save_btn = QPushButton("💾 Save"); save_btn.setObjectName("primaryBtn")
        save_btn.clicked.connect(self._save_labels); sl.addWidget(save_btn)

        sl.addStretch()
        sidebar_scroll.setWidget(sidebar)
        body.addWidget(sidebar_scroll)

        body.setStretchFactor(0, 0); body.setStretchFactor(1, 1); body.setStretchFactor(2, 0)
        root.addWidget(body, 1)

        # ======= Status bar =======
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar(); self.progress_bar.setFixedWidth(120)
        self.progress_bar.setRange(0, 0); self.progress_bar.hide()
        sb = self.statusBar()
        sb.addWidget(self.status_label, 1); sb.addPermanentWidget(self.progress_bar)

        # ======= Shortcuts =======
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, self._prev_image)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self._next_image)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, self._delete_selected)
        QShortcut(QKeySequence("Ctrl+S"), self, self._save_labels)
        QShortcut(QKeySequence("1"), self, lambda: self._set_mode(0))
        QShortcut(QKeySequence("2"), self, lambda: self._set_mode(1))
        QShortcut(QKeySequence("3"), self, lambda: self._set_mode(2))
        QShortcut(QKeySequence("Ctrl+A"), self, self._select_all)
        QShortcut(QKeySequence("Escape"), self, self._deselect_all)
        QShortcut(QKeySequence("F"), self, self.canvas.fit_view)

        # ======= Style =======
        self.setStyleSheet("""
            * { font-family: "Segoe UI", "Helvetica Neue", sans-serif; }
            QMainWindow, QWidget { background: #0f0f1a; color: #ccd6f6; }
            #navBar { background: #161625; border: 1px solid #1e2d4a; border-radius: 8px; }
            #toolPanel { background: #161625; border: 1px solid #1e2d4a; border-radius: 8px; }
            QPushButton {
                background: #1e2d4a; border: 1px solid #2a3f6f; border-radius: 5px;
                padding: 5px 10px; color: #ccd6f6; font-size: 12px;
            }
            QPushButton:hover { background: #2a3f6f; border-color: #4a6fa5; }
            QPushButton:pressed { background: #3a5f9f; }
            QPushButton#primaryBtn {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #5b6abf,stop:1 #7c5cbf);
                border: none; color: #fff; font-weight: 600;
            }
            QPushButton#primaryBtn:hover {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #6b7acf,stop:1 #8c6ccf);
            }
            QLineEdit, QComboBox {
                background: #1a1a2e; border: 1px solid #2a3f6f; border-radius: 4px;
                padding: 4px 8px; color: #ccd6f6; selection-background-color: #5b6abf;
            }
            QLineEdit:focus, QComboBox:focus { border-color: #5b6abf; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #1a1a2e; border: 1px solid #2a3f6f; color: #ccd6f6;
                selection-background-color: #5b6abf;
            }
            QListWidget {
                background: #12121f; border: 1px solid #1e2d4a; border-radius: 4px; color: #ccd6f6;
            }
            QListWidget::item { padding: 3px 6px; border-radius: 3px; }
            QListWidget::item:selected { background: #2a3f6f; }
            QListWidget::item:hover { background: #1e2d4a; }
            QRadioButton, QCheckBox { color: #8892b0; spacing: 8px; }
            QRadioButton::indicator, QCheckBox::indicator {
                width: 18px; height: 18px;
            }
            QSlider::groove:horizontal {
                height: 4px; background: #1e2d4a; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #5b6abf; width: 14px; height: 14px; margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal { background: #5b6abf; border-radius: 2px; }
            QProgressBar {
                background: #1a1a2e; border: 1px solid #2a3f6f; border-radius: 4px;
                text-align: center; color: #ccd6f6; font-size: 10px;
            }
            QProgressBar::chunk { background: #5b6abf; border-radius: 3px; }
            QStatusBar { background: #0f0f1a; color: #8892b0; border-top: 1px solid #1e2d4a; }
            QScrollArea { border: none; }
            QScrollBar:vertical {
                background: #12121f; width: 8px; border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #2a3f6f; min-height: 30px; border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover { background: #5b6abf; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QSplitter::handle { background: #1e2d4a; width: 2px; }
        """)

    # ==================================================================
    # Folder / image management
    # ==================================================================

    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select image folder")
        if d: self.folder_input.setText(d)

    def _load_folder(self):
        fp = self.folder_input.text().strip()
        op = self.output_input.text().strip() or "output"
        if not fp: self._set_status("Please enter a folder path"); return
        exts = {'.jpg','.jpeg','.png','.bmp','.webp'}
        imgs = sorted(p for p in Path(fp).iterdir() if p.suffix.lower() in exts)
        if not imgs: self._set_status("No images found in folder"); return
        self.state.image_list = imgs
        self.state.output_folder = Path(op)
        save_config(fp, op)
        last = load_progress(fp)
        if last >= len(imgs): last = 0
        self.state.current_index = last
        self._load_current_image()

    def _load_current_image(self):
        if not self.state.image_list: return
        ip = self.state.image_list[self.state.current_index]
        img = cv2.imread(str(ip))
        if img is None: self._set_status(f"Cannot read: {ip.name}"); return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.state.current_image = rgb
        self.state.current_image_path = ip
        self.state.current_labels = []
        self.state.selected_labels.clear()

        lp = self.state.output_folder / "labels" / f"{ip.stem}.txt"
        sp = self.state.output_folder / "labels_seg" / f"{ip.stem}.txt"
        self.state.current_labels = load_existing_labels(lp, sp, rgb)

        self.canvas.set_image(rgb)
        self._refresh_labels_ui()
        n = len(self.state.image_list); ci = self.state.current_index+1
        nl = len(self.state.current_labels)
        self._set_status(f"📷 {ci}/{n}  🏷️ {nl} annotations  |  {ip.name}")
        self.nav_label.setText(f"{ci} / {n}")
        self.jump_input.setText(str(ci))

    def _nav(self, delta):
        if not self.state.image_list: return
        self.state.output_folder = Path(self.output_input.text().strip() or "output")
        auto_save_labels(self.state)
        ni = self.state.current_index + delta
        if 0 <= ni < len(self.state.image_list):
            self.state.current_index = ni
            save_progress(self.state.image_list[0].parent, ni, self.state.image_list)
            self._load_current_image()

    def _prev_image(self): self._nav(-1)
    def _next_image(self): self._nav(1)

    def _jump_to(self):
        try: idx = int(self.jump_input.text())-1
        except ValueError: return
        if not self.state.image_list or not (0<=idx<len(self.state.image_list)): return
        self.state.output_folder = Path(self.output_input.text().strip() or "output")
        auto_save_labels(self.state)
        self.state.current_index = idx
        save_progress(self.state.image_list[0].parent, idx, self.state.image_list)
        self._load_current_image()

    # ==================================================================
    # SAM segmentation (background thread)
    # ==================================================================

    def _get_class_id(self):
        c = self.class_combo.currentText()
        return self.state.classes.index(c) if c in self.state.classes else 0

    def _start_busy(self):
        self.canvas.set_busy(True); self.progress_bar.show()

    def _end_busy(self):
        self.canvas.set_busy(False); self.progress_bar.hide()

    def _segment_text(self):
        if self.state.current_image is None: return
        prompts = [p.strip() for p in self.text_prompt.text().split(',') if p.strip()]
        if not prompts: return
        self._start_busy(); self._set_status("Running text segmentation...")

        def _do():
            return self.sam.segment_text(
                self.state.current_image, prompts, self.state.classes,
                self.state.current_labels, self.state.polygon_epsilon,
                self.state.overlap_threshold)

        self._worker = SAMWorker(_do)
        self._worker.finished.connect(self._on_text_seg_done)
        self._worker.start()

    def _on_text_seg_done(self, result, err):
        self._end_busy()
        if err: self._set_status(f"Error: {err}"); return
        if result is None: self._set_status("No results"); return
        new_labels, added, skipped, new_classes = result
        if new_classes:
            self.state.classes.extend(new_classes)
            persist_classes(self.state.classes); self._refresh_class_combos()
        self.state.current_labels.extend(new_labels)
        self._refresh_labels_ui()
        self._set_status(f"Detection complete: added {added}, skipped {skipped}")

    def _on_point_click(self, x, y):
        if self.state.current_image is None: return
        cid = self._get_class_id()
        self._start_busy(); self._set_status(f"Segmenting ({x},{y})...")

        def _do():
            return self.sam.segment_point(
                self.state.current_image, x, y, cid,
                self.state.current_labels, self.state.polygon_epsilon,
                self.state.overlap_threshold)

        self._worker = SAMWorker(_do)
        self._worker.finished.connect(self._on_point_seg_done)
        self._worker.start()

    def _on_point_seg_done(self, result, err):
        self._end_busy()
        if err: self._set_status(f"Error: {err}"); return
        if result is None: self._set_status("No results"); return
        label, msg = result
        if label:
            self.state.current_labels.append(label)
            self._refresh_labels_ui()
        self._set_status(msg)

    def _on_box_drawn(self, x1, y1, x2, y2):
        if self.state.current_image is None: return
        cid = self._get_class_id()
        self._start_busy(); self._set_status("Box segmenting...")

        def _do():
            return self.sam.segment_box(
                self.state.current_image, x1, y1, x2, y2, cid,
                self.state.current_labels, self.state.polygon_epsilon,
                self.state.overlap_threshold, self.fallback_cb.isChecked())

        self._worker = SAMWorker(_do)
        self._worker.finished.connect(self._on_point_seg_done)  # same structure
        self._worker.start()

    # ==================================================================
    # UI sync
    # ==================================================================

    def _set_status(self, msg):
        self.status_label.setText(msg)

    def _refresh_labels_ui(self):
        self.canvas.set_labels(self.state.current_labels, self.state.classes, self.state.selected_labels)
        self.label_list.blockSignals(True); self.label_list.clear()
        for idx, lb in enumerate(self.state.current_labels):
            cid = lb[0]
            cn = self.state.classes[cid] if cid<len(self.state.classes) else f"c{cid}"
            item = QListWidgetItem(f"  {idx+1}. {cn}")
            item.setIcon(color_icon(LABEL_COLORS[cid%len(LABEL_COLORS)]))
            self.label_list.addItem(item)
        self.label_list.blockSignals(False)

    def _refresh_class_combos(self):
        for cb in (self.class_combo, self.change_combo):
            cur = cb.currentText(); cb.clear(); cb.addItems(self.state.classes)
            if cur in self.state.classes: cb.setCurrentText(cur)

    def _on_label_selected(self, idx):
        self.state.selected_labels = self.canvas._selected.copy()
        self.label_list.blockSignals(True); self.label_list.clearSelection()
        for i in self.state.selected_labels:
            if i < self.label_list.count(): self.label_list.item(i).setSelected(True)
        self.label_list.blockSignals(False)
        n = len(self.state.selected_labels)
        if n: self._set_status(f"{n} annotations selected")

    def _on_list_selection(self):
        sel = set()
        for it in self.label_list.selectedItems():
            try: sel.add(int(it.text().strip().split('.')[0])-1)
            except ValueError: pass
        self.state.selected_labels = sel
        self.canvas.set_selected(sel)

    def _mode_changed(self, btn):
        self.canvas.set_mode(btn.property("mode"))

    def _set_mode(self, idx):
        btn = self.mode_group.button(idx)
        if btn: btn.setChecked(True); self.canvas.set_mode(btn.property("mode"))

    def _display_mode_changed(self, btn):
        dm = btn.property("dm"); self.state.display_mode = dm; self.canvas.set_display_mode(dm)

    def _epsilon_changed(self, val):
        v = val / 1000.0; self.state.polygon_epsilon = v; self.epsilon_label.setText(f"{v:.3f}")

    def _overlap_changed(self, val):
        v = val / 100.0; self.state.overlap_threshold = v
        self.overlap_label.setText("Off" if val==0 else f"{val}%")

    def _fmt_changed(self):
        self.state.output_formats["obb"] = self.fmt_obb.isChecked()
        self.state.output_formats["seg"] = self.fmt_seg.isChecked()
        self.state.output_formats["mask"] = self.fmt_mask.isChecked()
        self.state.output_formats["coco"] = self.fmt_coco.isChecked()

    # ==================================================================
    # Annotation operations
    # ==================================================================

    def _delete_selected(self):
        if not self.state.selected_labels: return
        for i in sorted(self.state.selected_labels, reverse=True):
            if i < len(self.state.current_labels): del self.state.current_labels[i]
        self.state.selected_labels.clear()
        self._refresh_labels_ui(); self._set_status("Deleted selected annotations")

    def _clear_all(self):
        self.state.current_labels.clear(); self.state.selected_labels.clear()
        self._refresh_labels_ui(); self._set_status("Cleared all annotations")

    def _select_all(self):
        self.state.selected_labels = set(range(len(self.state.current_labels)))
        self.canvas.set_selected(self.state.selected_labels)
        self.label_list.selectAll()

    def _deselect_all(self):
        self.state.selected_labels.clear(); self.canvas.set_selected(set())
        self.label_list.clearSelection()

    def _change_selected_class(self):
        nc = self.change_combo.currentText()
        if not nc or nc not in self.state.classes: return
        nid = self.state.classes.index(nc)
        for i in self.state.selected_labels:
            if i < len(self.state.current_labels):
                lb = self.state.current_labels[i]
                self.state.current_labels[i] = (nid,)+lb[1:]
        self._refresh_labels_ui(); self._set_status(f"Changed to {nc}")

    def _add_class(self):
        n = self.new_class_input.text().strip()
        if not n or n in self.state.classes: return
        self.state.classes.append(n)
        persist_classes(self.state.classes); self._refresh_class_combos()
        self.new_class_input.clear(); self._set_status(f"Added class: {n}")

    def _delete_class(self):
        c = self.class_combo.currentText()
        if not c or len(self.state.classes)<=1: return
        cid = self.state.classes.index(c)
        using = sum(1 for lb in self.state.current_labels if lb[0]==cid)
        if using: self._set_status(f"{using} annotations use this class, cannot delete"); return
        self.state.classes.remove(c)
        persist_classes(self.state.classes); self._refresh_class_combos()
        self._set_status(f"Deleted class: {c}")

    def _save_labels(self):
        self.state.output_folder = Path(self.output_input.text().strip() or "output")
        msg = auto_save_labels(self.state)
        self._set_status(msg or "Saved successfully")
