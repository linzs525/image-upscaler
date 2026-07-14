"""
gui.py — PySide6 图形界面

布局概览:
┌──────────────────────────────────────────────────┐
│  菜单栏: [文件] [设置] [帮助]                      │
├────────────┬─────────────────────────────────────┤
│  文件列表   │  预览区域 (前 / 后对比)               │
│  (拖拽添加) │                                     │
│            │  ┌─────────┐  ┌─────────┐           │
│  📄 cat    │  │  原图    │  │  结果   │           │
│  📄 dog    │  │          │  │         │           │
│            │  └─────────┘  └─────────┘           │
├────────────┴─────────────────────────────────────┤
│  设置栏: 模型 [▼]  倍数 [▼]  格式 [▼]  设备 [▼]   │
│  [选择图片]  [开始处理]  [批量处理]                │
│  ████████████░░░░░░░░░░  45%                     │
└──────────────────────────────────────────────────┘
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QPushButton,
    QComboBox,
    QProgressBar,
    QStatusBar,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QScrollArea,
    QTabWidget,
)
from PySide6.QtCore import (
    Qt,
    QThread,
    Signal,
)
from PySide6.QtGui import (
    QPixmap,
    QImage,
    QDragEnterEvent,
    QDropEvent,
    QAction,
    QFont,
)

from .config import (
    MODELS,
    DEFAULT_MODEL,
    DEFAULT_SCALE,
    AVAILABLE_SCALES,
    SUPPORTED_OUTPUT_FORMATS,
    SUPPORTED_INPUT_FORMATS,
    WINDOW_TITLE,
    WINDOW_MIN_WIDTH,
    WINDOW_MIN_HEIGHT,
    PREVIEW_MAX_SIZE,
    OUTPUT_DIR,
)
from .utils import (
    bgr_to_rgb,
    load_image,
    is_supported_input,
    get_image_info,
    ensure_dir,
    resize_bgr,
)


# ================================================================
# 后台处理线程
# ================================================================

class UpscaleWorker(QThread):
    """在后台线程加载模型并执行超分，避免阻塞 GUI。"""
    ready = Signal(object, str)       # (upscaler, backend_message)
    file_started = Signal(str)        # input_path
    file_done = Signal(str, str)      # (input_path, output_path)
    file_failed = Signal(str, str)    # (input_path, error_message)
    progressing = Signal(int, int)    # (current, total)
    status = Signal(str)
    completed = Signal(int, int)      # (success_count, failed_count)
    canceled = Signal()

    def __init__(
        self,
        upscaler,
        upscaler_options,
        input_paths,
        output_dir,
        output_format,
    ):
        super().__init__()
        self._upscaler = upscaler
        self._upscaler_options = upscaler_options
        self._input_paths = input_paths
        self._output_dir = output_dir
        self._output_format = output_format
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        success_count = 0
        failed_count = 0
        total = len(self._input_paths)

        try:
            if self._upscaler is None:
                from .upscaler import Upscaler

                model_name, scale, device = self._upscaler_options
                self.status.emit("正在加载模型，请稍候...")
                self._upscaler = Upscaler(
                    model_name=model_name,
                    scale=scale,
                    device=device,
                )

            backend_message = (
                f"后端: {self._upscaler.backend_name}，"
                f"设备: {self._upscaler.device.upper()}"
            )
            if self._upscaler.fallback_reason:
                backend_message += "（Real-ESRGAN 不可用，使用高质量插值）"
            self.ready.emit(self._upscaler, backend_message)
        except Exception as e:
            self.file_failed.emit("", f"模型加载失败: {e}")
            self.completed.emit(0, total)
            return

        self.progressing.emit(0, total)
        for idx, path in enumerate(self._input_paths, 1):
            if self._cancel_requested:
                self.canceled.emit()
                break

            self.file_started.emit(str(path))
            self.status.emit(f"正在处理 {idx}/{total}: {Path(path).name}")
            try:
                out = self._upscaler.upscale(
                    path,
                    output_dir=self._output_dir,
                    output_format=self._output_format,
                )
                success_count += 1
                self.file_done.emit(str(path), str(out))
            except Exception as e:
                failed_count += 1
                self.file_failed.emit(str(path), str(e))
            self.progressing.emit(idx, total)

        self.completed.emit(success_count, failed_count)


# ================================================================
# 拖拽区域
# ================================================================

class DropZone(QLabel):
    """支持拖拽图片文件到列表的区域"""
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setText("拖拽图片文件到此处\n或点击下方 [选择图片] 按钮")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(80)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #888;
                border-radius: 8px;
                padding: 12px;
                color: #999;
                font-size: 13px;
                background: #fafafa;
            }
        """)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                QLabel {
                    border: 2px dashed #4a90d9;
                    border-radius: 8px;
                    padding: 12px;
                    color: #4a90d9;
                    font-size: 13px;
                    background: #e8f0fe;
                }
            """)

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #888;
                border-radius: 8px;
                padding: 12px;
                color: #999;
                font-size: 13px;
                background: #fafafa;
            }
        """)

    def dropEvent(self, event: QDropEvent):
        self.dragLeaveEvent(None)
        paths = []
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_dir():
                paths.extend(
                    str(file)
                    for file in sorted(p.rglob("*"))
                    if file.is_file() and is_supported_input(file)
                )
            elif is_supported_input(p):
                paths.append(str(p))
        if paths:
            self.files_dropped.emit(paths)


# ================================================================
# 预览组件
# ================================================================

class PreviewPanel(QWidget):
    """原图 / 结果 预览面板"""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        self._original_label = QLabel("尚未选择图片")
        self._original_label.setAlignment(Qt.AlignCenter)
        self._original_label.setStyleSheet("color: #999;")

        self._result_label = QLabel("处理完成后显示结果")
        self._result_label.setAlignment(Qt.AlignCenter)
        self._result_label.setStyleSheet("color: #999;")

        orig_scroll = QScrollArea()
        orig_scroll.setWidgetResizable(True)
        orig_scroll.setWidget(self._original_label)

        result_scroll = QScrollArea()
        result_scroll.setWidgetResizable(True)
        result_scroll.setWidget(self._result_label)

        self._tabs.addTab(orig_scroll, "📷 原图")
        self._tabs.addTab(result_scroll, "✨ 结果")

        layout.addWidget(self._tabs)

    def show_original(self, filepath: str):
        """显示原图预览"""
        pixmap = self._load_pixmap(filepath)
        if pixmap:
            self._original_label.setPixmap(pixmap)
            self._original_label.setText("")
        else:
            self._original_label.setText("无法加载图片")

    def show_result(self, filepath: str):
        """显示结果预览并自动切换到结果标签"""
        pixmap = self._load_pixmap(filepath)
        if pixmap:
            self._result_label.setPixmap(pixmap)
            self._result_label.setText("")
        else:
            self._result_label.setText("无法加载结果")
        self._tabs.setCurrentIndex(1)

    def clear_preview(self):
        self._original_label.setText("尚未选择图片")
        self._original_label.setPixmap(QPixmap())
        self._result_label.setText("处理完成后显示结果")
        self._result_label.setPixmap(QPixmap())

    @staticmethod
    def _load_pixmap(filepath: str) -> QPixmap | None:
        """安全加载并缩放到预览尺寸"""
        try:
            img = load_image(filepath)
            h, w = img.shape[:2]
            max_w, max_h = PREVIEW_MAX_SIZE
            scale = min(max_w / w, max_h / h, 1.0)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                img = resize_bgr(img, (new_w, new_h))

            # BGR → RGB QImage
            rgb = bgr_to_rgb(img)
            h, w, c = rgb.shape
            qimg = QImage(rgb.data, w, h, w * c, QImage.Format_RGB888)
            return QPixmap.fromImage(qimg.copy())
        except Exception:
            return None


# ================================================================
# 主窗口
# ================================================================

class MainWindow(QMainWindow):
    """图片高清化工具 — 主窗口"""

    def __init__(self, upscaler=None):
        super().__init__()
        self._upscaler = upscaler  # 外部传入或稍后创建
        self._file_map: dict[str, dict] = {}  # path → {item, info}
        self._worker: UpscaleWorker | None = None
        self._upscaler_options: tuple[str, int, str] | None = None
        self._done_count = 0
        self._total_count = 0
        self._was_canceled = False

        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self._setup_menubar()
        self._setup_ui()
        self._setup_statusbar()

    # ------------------------------------------------------------------
    # 菜单栏
    # ------------------------------------------------------------------
    def _setup_menubar(self):
        mb = self.menuBar()

        # 文件菜单
        file_menu = mb.addMenu("文件(&F)")
        act_open = QAction("选择图片(&O)...", self)
        act_open.triggered.connect(self._on_select_files)
        file_menu.addAction(act_open)

        act_clear = QAction("清空列表(&C)", self)
        act_clear.triggered.connect(self._on_clear_list)
        file_menu.addAction(act_clear)

        file_menu.addSeparator()

        act_exit = QAction("退出(&Q)", self)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # 设置菜单
        settings_menu = mb.addMenu("设置(&S)")
        act_outdir = QAction("设置输出目录...", self)
        act_outdir.triggered.connect(self._on_set_output_dir)
        settings_menu.addAction(act_outdir)

        # 帮助菜单
        help_menu = mb.addMenu("帮助(&H)")
        act_about = QAction("关于(&A)", self)
        act_about.triggered.connect(self._on_about)
        help_menu.addAction(act_about)

    # ------------------------------------------------------------------
    # 主界面
    # ------------------------------------------------------------------
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ---- 中部：文件列表 + 预览 ----
        splitter = QSplitter(Qt.Horizontal)

        # 左侧面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 拖拽区域
        self._drop_zone = DropZone()
        self._drop_zone.files_dropped.connect(self._on_files_dropped)
        left_layout.addWidget(self._drop_zone)

        # 文件列表
        self._file_list = QListWidget()
        self._file_list.currentItemChanged.connect(self._on_file_selected)
        left_layout.addWidget(self._file_list)

        splitter.addWidget(left_panel)

        # 右侧预览
        self._preview = PreviewPanel()
        splitter.addWidget(self._preview)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, stretch=1)

        # ---- 底部：设置栏 ----
        settings_group = QGroupBox("处理设置")
        settings_layout = QHBoxLayout(settings_group)

        # 模型
        settings_layout.addWidget(QLabel("模型:"))
        self._combo_model = QComboBox()
        for key, display in MODELS.items():
            self._combo_model.addItem(display, key)
        self._combo_model.setCurrentIndex(
            list(MODELS.keys()).index(DEFAULT_MODEL)
        )
        self._combo_model.currentIndexChanged.connect(self._on_settings_changed)
        settings_layout.addWidget(self._combo_model)

        # 倍数
        settings_layout.addWidget(QLabel("倍数:"))
        self._combo_scale = QComboBox()
        for s in AVAILABLE_SCALES:
            self._combo_scale.addItem(f"{s}x", s)
        self._combo_scale.setCurrentIndex(
            AVAILABLE_SCALES.index(DEFAULT_SCALE)
        )
        self._combo_scale.currentIndexChanged.connect(self._on_settings_changed)
        settings_layout.addWidget(self._combo_scale)

        # 输出格式
        settings_layout.addWidget(QLabel("格式:"))
        self._combo_format = QComboBox()
        for fmt in SUPPORTED_OUTPUT_FORMATS:
            self._combo_format.addItem(fmt, fmt)
        self._combo_format.setCurrentIndex(0)  # .png
        settings_layout.addWidget(self._combo_format)

        # 设备
        settings_layout.addWidget(QLabel("设备:"))
        self._combo_device = QComboBox()
        self._combo_device.addItem("自动", "auto")
        self._combo_device.addItem("CPU", "cpu")
        if _cuda_available():
            self._combo_device.addItem("CUDA (GPU)", "cuda")
        self._combo_device.currentIndexChanged.connect(self._on_settings_changed)
        settings_layout.addWidget(self._combo_device)

        settings_layout.addStretch()

        root.addWidget(settings_group)

        # ---- 按钮行 ----
        btn_layout = QHBoxLayout()

        self._btn_select = QPushButton("选择图片")
        self._btn_select.clicked.connect(self._on_select_files)
        btn_layout.addWidget(self._btn_select)

        self._btn_start = QPushButton("开始处理")
        self._btn_start.clicked.connect(self._on_start)
        self._btn_start.setStyleSheet("""
            QPushButton {
                background: #4a90d9; color: white; font-weight: bold;
                padding: 6px 20px; border-radius: 4px;
            }
            QPushButton:hover { background: #3a7bc8; }
            QPushButton:disabled { background: #ccc; }
        """)
        btn_layout.addWidget(self._btn_start)

        self._btn_cancel = QPushButton("取消")
        self._btn_cancel.clicked.connect(self._on_cancel)
        self._btn_cancel.setEnabled(False)
        btn_layout.addWidget(self._btn_cancel)

        self._btn_clear = QPushButton("清空列表")
        self._btn_clear.clicked.connect(self._on_clear_list)
        btn_layout.addWidget(self._btn_clear)

        self._btn_outdir = QPushButton("输出目录")
        self._btn_outdir.clicked.connect(self._on_set_output_dir)
        btn_layout.addWidget(self._btn_outdir)

        btn_layout.addStretch()
        root.addLayout(btn_layout)

        # ---- 进度条 ----
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        # 输出目录
        self._output_dir = str(OUTPUT_DIR)
        self._lbl_outdir = QLabel(f"输出目录: {self._output_dir}")
        self._lbl_outdir.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._lbl_outdir.setStyleSheet("color: #888; font-size: 11px;")
        root.addWidget(self._lbl_outdir)

    # ------------------------------------------------------------------
    # 状态栏
    # ------------------------------------------------------------------
    def _setup_statusbar(self):
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("就绪 — 请拖入图片或点击 [选择图片] 开始")

    # ------------------------------------------------------------------
    # 槽函数 — 文件操作
    # ------------------------------------------------------------------
    def _on_select_files(self):
        """打开文件选择对话框"""
        filter_str = "图片文件 ({});;所有文件 (*.*)".format(
            " ".join(f"*{ext}" for ext in SUPPORTED_INPUT_FORMATS)
        )
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "", filter_str
        )
        if paths:
            self._add_files(paths)

    def _on_files_dropped(self, paths: list[str]):
        """拖拽文件回调"""
        self._add_files(paths)

    def _add_files(self, paths: list[str]):
        """向列表添加文件（去重）"""
        added = 0
        for p in paths:
            p = str(Path(p).resolve())
            if p in self._file_map:
                continue
            if not is_supported_input(p):
                continue
            try:
                info = get_image_info(p)
            except Exception:
                self._status.showMessage(f"无法读取图片: {p}")
                continue

            item = QListWidgetItem()
            item.setData(Qt.UserRole, p)
            item.setToolTip(p)
            self._file_list.addItem(item)
            self._file_map[p] = {"item": item, "info": info, "status": "等待"}
            self._set_file_status(p, "等待")
            added += 1

        if added > 0:
            self._status.showMessage(f"已添加 {added} 张图片，共 {len(self._file_map)} 张")
            if self._file_list.count() == added:
                # 自动选中第一张预览
                self._file_list.setCurrentRow(0)

    def _on_clear_list(self):
        self._file_list.clear()
        self._file_map.clear()
        self._preview.clear_preview()
        self._status.showMessage("列表已清空")

    def _on_file_selected(self, current, previous):
        """选中文件时预览原图"""
        if current is None:
            return
        path = current.data(Qt.UserRole)
        if path:
            self._preview.show_original(path)
            self._status.showMessage(
                f"选中: {Path(path).name}  "
                f"({self._file_map[path]['info']['width']}×"
                f"{self._file_map[path]['info']['height']})"
            )

    # ------------------------------------------------------------------
    # 槽函数 — 处理
    # ------------------------------------------------------------------
    def _on_start(self):
        """开始处理"""
        if not self._file_map:
            QMessageBox.information(self, "提示", "请先添加图片文件。")
            return

        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "提示", "正在处理中，请等待完成。")
            return

        selected_options = self._selected_upscaler_options()
        upscaler = self._upscaler if self._upscaler_options == selected_options else None

        # 收集输入
        input_paths = list(self._file_map.keys())
        output_format = self._combo_format.currentData()
        ensure_dir(self._output_dir)

        for path in input_paths:
            self._set_file_status(path, "等待")

        self._set_processing_ui(True)

        # 进度条
        self._progress.setVisible(True)
        self._progress.setMaximum(len(input_paths))
        self._progress.setValue(0)

        # 启动后台线程
        self._done_count = 0
        self._total_count = len(input_paths)
        self._was_canceled = False

        self._worker = UpscaleWorker(
            upscaler,
            selected_options,
            input_paths,
            self._output_dir,
            output_format,
        )
        self._worker.ready.connect(self._on_worker_ready)
        self._worker.file_started.connect(self._on_file_started)
        self._worker.file_done.connect(self._on_one_done)
        self._worker.file_failed.connect(self._on_one_failed)
        self._worker.file_failed.connect(self._on_file_finished)
        self._worker.progressing.connect(self._on_progress)
        self._worker.status.connect(self._status.showMessage)
        self._worker.file_done.connect(self._on_file_finished)
        self._worker.completed.connect(self._on_completed)
        self._worker.canceled.connect(self._on_canceled)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_progress(self, current, total):
        self._progress.setValue(current)

    def _on_worker_ready(self, upscaler, backend_message):
        self._upscaler = upscaler
        self._upscaler_options = self._selected_upscaler_options()
        self._status.showMessage(f"模型已就绪，{backend_message}")

    def _on_file_started(self, input_path):
        self._set_file_status(input_path, "处理中")

    def _on_one_done(self, input_path, output_path):
        self._set_file_status(input_path, "完成", output_path=output_path)
        self._status.showMessage(f"完成: {Path(output_path).name}")
        # 预览结果（最后处理的那张）
        self._preview.show_result(output_path)

    def _on_one_failed(self, input_path, error):
        if input_path:
            self._set_file_status(input_path, "失败", error=error)
            self._status.showMessage(f"失败: {Path(input_path).name} - {error}")
        else:
            self._status.showMessage(error)
            QMessageBox.critical(self, "处理失败", error)

    def _on_file_finished(self, *args):
        """每个文件结束时检查是否全部完成，若是则恢复 UI"""
        self._done_count += 1
        if args:
            path = args[0]
            if path in self._file_map and self._file_map[path]["status"] == "等待":
                self._set_file_status(path, "处理中")

    def _on_completed(self, success_count, failed_count):
        self._set_processing_ui(False)
        self._progress.setVisible(False)
        self._worker = None
        if self._was_canceled:
            message = (
                f"已取消: 已完成 {success_count} 张，失败 {failed_count} 张。"
                f"输出目录: {self._output_dir}"
            )
        else:
            message = (
                f"处理完成: 成功 {success_count} 张，失败 {failed_count} 张。"
                f"输出目录: {self._output_dir}"
            )
        self._status.showMessage(message, timeout=8000)

    def _on_canceled(self):
        self._was_canceled = True
        self._status.showMessage("已取消，当前图片处理结束后停止。")

    def _on_cancel(self):
        if self._worker and self._worker.isRunning():
            self._btn_cancel.setEnabled(False)
            self._status.showMessage("正在取消，请稍候...")
            self._worker.cancel()

    def _set_file_status(
        self,
        path: str,
        status: str,
        output_path: str | None = None,
        error: str | None = None,
    ):
        if path not in self._file_map:
            return

        entry = self._file_map[path]
        entry["status"] = status
        if output_path:
            entry["output_path"] = output_path
        if error:
            entry["error"] = error

        item = entry["item"]
        info = entry["info"]
        item.setText(
            f"[{status}] {info['filename']}  "
            f"({info['width']}x{info['height']}, {info['size_kb']} KB)"
        )

        details = path
        if output_path:
            details += f"\n输出: {output_path}"
        if error:
            details += f"\n错误: {error}"
        item.setToolTip(details)

        if status == "完成":
            item.setForeground(Qt.darkGreen)
        elif status == "失败":
            item.setForeground(Qt.red)
        elif status == "处理中":
            item.setForeground(Qt.blue)
        else:
            item.setForeground(Qt.black)

    def _set_processing_ui(self, is_processing: bool):
        self._btn_start.setEnabled(not is_processing)
        self._btn_start.setText("处理中..." if is_processing else "开始处理")
        self._btn_cancel.setEnabled(is_processing)
        self._btn_select.setEnabled(not is_processing)
        self._btn_clear.setEnabled(not is_processing)
        self._btn_outdir.setEnabled(not is_processing)
        self._combo_model.setEnabled(not is_processing)
        self._combo_scale.setEnabled(not is_processing)
        self._combo_format.setEnabled(not is_processing)
        self._combo_device.setEnabled(not is_processing)

    def _on_settings_changed(self):
        if self._worker and self._worker.isRunning():
            return
        self._upscaler = None
        self._upscaler_options = None
        self._status.showMessage("处理设置已更新，开始处理时将重新加载模型")

    # ------------------------------------------------------------------
    # 槽函数 — 菜单
    # ------------------------------------------------------------------
    def _on_set_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录", self._output_dir)
        if d:
            self._output_dir = d
            self._lbl_outdir.setText(f"输出目录: {self._output_dir}")

    def _on_about(self):
        QMessageBox.about(
            self,
            "关于 — 图片高清化工具",
            "<h3>图片高清化工具 v1.0</h3>"
            "<p>基于 <b>Real-ESRGAN</b> 深度学习模型的图片超分辨率应用。</p>"
            "<p>支持 PNG / JPG / BMP / TIFF / WebP 等常见格式。</p>"
            "<hr>"
            "<p>模型来源: xinntao/Real-ESRGAN (GitHub)</p>"
            "<p>GUI 框架: PySide6 (Qt for Python)</p>",
        )

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _init_upscaler(self):
        """延迟初始化 Upscaler（需要时间下载模型）"""
        from .upscaler import Upscaler

        model_key = self._combo_model.currentData()
        scale = self._combo_scale.currentData()
        device = self._combo_device.currentData()
        selected_options = (model_key, scale, device)

        self._status.showMessage("正在加载模型（首次使用将下载权重，请稍候）...")
        QApplication.processEvents()

        try:
            self._upscaler = Upscaler(
                model_name=model_key,
                scale=scale,
                device=device,
            )
            self._upscaler_options = selected_options
            backend_note = f"后端: {self._upscaler.backend_name}"
            if self._upscaler.fallback_reason:
                backend_note += "（未检测到 Real-ESRGAN 依赖，使用高质量插值）"
            self._status.showMessage(
                f"模型已就绪: {self._upscaler.model_display_name}  "
                f"设备: {self._upscaler.device.upper()}  {backend_note}",
                timeout=5000,
            )
        except Exception as e:
            QMessageBox.critical(self, "模型加载失败", str(e))
            self._status.showMessage("模型加载失败")
            self._upscaler = None
            self._upscaler_options = None

    def set_upscaler(self, upscaler):
        """外部注入已初始化的 Upscaler"""
        self._upscaler = upscaler
        self._upscaler_options = (
            upscaler.model_name,
            upscaler.scale,
            upscaler.device,
        )
        self._status.showMessage(
            f"模型已就绪: {upscaler.model_display_name}  "
            f"设备: {upscaler.device.upper()}  后端: {upscaler.backend_name}",
            timeout=5000,
        )

    def _selected_upscaler_options(self) -> tuple[str, int, str]:
        return (
            self._combo_model.currentData(),
            self._combo_scale.currentData(),
            self._combo_device.currentData(),
        )


def _cuda_available() -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return torch.cuda.is_available()


# ================================================================
# 启动入口
# ================================================================

def launch(upscaler=None):
    """启动 GUI 应用。可传入已初始化的 Upscaler 实例。"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 全局字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    # 暗色主题调色板
    # （保留 Fusion 默认亮色，简洁清晰）

    window = MainWindow(upscaler=upscaler)
    window.show()
    sys.exit(app.exec())
