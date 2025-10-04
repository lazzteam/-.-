
#!/usr/bin/env python3
"""
Уникализатор Lazz.org — Artist Video Enhancer (GUI)

Этот файл — расширение предыдущего уникализатора: теперь в нём есть
набор художественных и технических трансформаций для видео и изображений:
- вставка/удаление кадров (тонко, незаметно для зрителя)
- микроповорот (wiggle) ±0.2° с кропом
- hue / saturation shift
- прозрачный шумовой слой (watermark noise)
- аудио-фильтры: highpass, fade-in/out, простая эквализация через ffmpeg
- дополнительные пресеты (мягкий / средний / жёсткий)

Важно: этот инструмент создан для **художественной обработки и улучшения**
видео/фото, а не для обхода систем распознавания или нарушения авторских прав.

Зависимости:
  - Python 3.9+
  - ffmpeg (в PATH)
  - pip install -r requirements (PySide6 Pillow numpy opencv-python)

Запуск:
  python media_uniq_app.py

"""

import sys
import os
import random
import subprocess
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QFileDialog, QLabel, QProgressBar,
    QSpinBox, QComboBox, QLineEdit, QMessageBox, QCheckBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPalette, QColor

SUPPORTED_VIDEO = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
SUPPORTED_IMAGE = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


# ----------------------- Helper functions -----------------------

def run_cmd(cmd):
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def extract_audio(src_path, out_audio_path):
    cmd = ['ffmpeg', '-y', '-i', str(src_path), '-vn', '-acodec', 'copy', str(out_audio_path)]
    return run_cmd(cmd)


def mux_audio_video(video_path, audio_path, out_path, crf=23):
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path), '-i', str(audio_path),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', str(crf),
        '-c:a', 'aac', '-b:a', '128k',
        str(out_path)
    ]
    return run_cmd(cmd)


# ----------------------- Video utilities (OpenCV) -----------------------

def load_video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError('Cannot open video')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = frame_count / fps if fps else 0
    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration
    }


def insert_and_remove_frames_opencv(src, dst, every_seconds=5, insert_count=1, remove_prob=0.0,
                                    micro_rotate_deg=0.0, hue_shift=0.0, sat_mult=1.0,
                                    watermark_noise_strength=0.0):
    """
    Читает src, применяет пост-обработку по кадрам и записывает dst.
    - every_seconds: через сколько секунд пытаться вставить кадр
    - insert_count: сколько дублирующих кадров вставлять
    - remove_prob: вероятность удалить текущий кадр (0..1)
    - micro_rotate_deg: случайный поворот в диапазоне ±micro_rotate_deg (в градусах)
    - hue_shift: смещение оттенка в диапазоне [-0.05, 0.05]
    - sat_mult: множитель насыщенности
    - watermark_noise_strength: (0..1) — слабый шумовой слой
    """
    info = load_video_info(src)
    cap = cv2.VideoCapture(str(src))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tmp_out = Path(dst).with_suffix('.tmp.mp4')
    out = cv2.VideoWriter(str(tmp_out), fourcc, info['fps'], (info['width'], info['height']))

    interval_frames = max(1, int(info['fps'] * max(0.1, every_seconds)))
    frame_idx = 0
    inserted = 0

    ret, frame = cap.read()
    while ret:
        # Решаем — удаляем ли кадр
        if remove_prob > 0 and random.random() < remove_prob:
            # пропускаем запись этого кадра
            ret, frame = cap.read()
            frame_idx += 1
            continue

        # Применяем микроповорот (случайный в маленьком диапазоне) + crop
        if micro_rotate_deg and abs(micro_rotate_deg) > 1e-6:
            angle = random.uniform(-micro_rotate_deg, micro_rotate_deg)
            # rotation
            h, w = frame.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            frame = rotated

        # Hue/Saturation tweak: convert BGR->HSV
        if abs(hue_shift) > 1e-6 or abs(sat_mult - 1.0) > 1e-6:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype('float32')
            # H channel: 0..179 in OpenCV
            hsv[..., 0] = (hsv[..., 0] + hue_shift * 179) % 179
            hsv[..., 1] = np.clip(hsv[..., 1] * sat_mult, 0, 255)
            frame = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2BGR)

        # Добавляем прозрачный шумовой слой (watermark noise)
        if watermark_noise_strength and watermark_noise_strength > 1e-6:
            noise = (np.random.randn(*frame.shape) * 255 * watermark_noise_strength).astype('int32')
            frame = np.clip(frame.astype('int32') + noise, 0, 255).astype('uint8')
            # слегка смешаем с оригиналом чтобы не переусердствовать
            frame = cv2.addWeighted(frame, 0.5, frame, 0.5, 0)

        # Записываем основной кадр
        out.write(frame)

        # Вставка дублей каждый interval_frames
        if frame_idx > 0 and frame_idx % interval_frames == 0 and insert_count > 0:
            for _ in range(insert_count):
                out.write(frame)
                inserted += 1

        frame_idx += 1
        ret, frame = cap.read()

    cap.release()
    out.release()

    # переименуем tmp в dst
    os.replace(str(tmp_out), str(dst))
    return True


# ----------------------- GUI / Worker -----------------------

class Worker(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished_all = Signal()

    def __init__(self, tasks, params):
        super().__init__()
        self.tasks = tasks
        self.params = params
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        total = len(self.tasks)
        for i, task in enumerate(self.tasks, start=1):
            if self._stopped:
                self.status.emit('Отменено')
                break

            src = Path(task)
            ext = src.suffix.lower()
            out_dir = Path(self.params['output_dir'])
            out_dir.mkdir(parents=True, exist_ok=True)

            if ext in SUPPORTED_VIDEO:
                self.status.emit(f'Обработка видео: {src.name}')
                ok = self.process_video(src, out_dir)
            elif ext in SUPPORTED_IMAGE:
                self.status.emit(f'Обработка изображения: {src.name}')
                ok = self.process_image(src, out_dir)
            else:
                self.status.emit(f'Пропуск (неподдерживаемый): {src.name}')
                ok = False

            pct = int(i / total * 100)
            self.progress.emit(pct)

        self.finished_all.emit()

    def process_video(self, src: Path, out_dir: Path) -> bool:
        params = self.params
        info = load_video_info(src)

        # временные файлы
        tmp_video = out_dir / f"tmp_{src.stem}.mp4"
        final_video = out_dir / f"enh_{src.stem}.mp4"
        tmp_audio = out_dir / f"tmp_{src.stem}.aac"

        # 1) Сначала делаем пост-обработку на уровне кадров (OpenCV): вставки/удаления/повороты/цвет/шум
        insert_every = params.get('insert_every', 5)
        insert_count = params.get('insert_count', 0)
        remove_prob = params.get('remove_prob', 0.0)
        micro_rotate = params.get('micro_rotate_deg', 0.2)
        hue_shift = params.get('hue_shift_pct', 0.0)  # в долях (напр. 0.02)
        sat_mult = params.get('saturation_mult', 1.0)
        watermark_noise = params.get('watermark_noise', 0.0)

        # Если не нужно модифицировать покадрово — просто копируем/перекодируем
        need_frame_ops = any([insert_count > 0, remove_prob > 0, abs(micro_rotate) > 1e-6,
                              abs(hue_shift) > 1e-6, abs(sat_mult - 1.0) > 1e-6, watermark_noise > 1e-6])

        if need_frame_ops:
            # используем OpenCV для создания временного видео (без аудио)
            self.status.emit('Покадровая обработка (OpenCV) ...')
            insert_and_remove_frames_opencv(str(src), str(tmp_video), every_seconds=insert_every,
                                            insert_count=insert_count, remove_prob=remove_prob,
                                            micro_rotate_deg=micro_rotate, hue_shift=hue_shift,
                                            sat_mult=sat_mult, watermark_noise_strength=watermark_noise)
        else:
            # просто копируем (перекодируем) исходник во временный файл
            self.status.emit('Копирование/перекодирование видео ...')
            cmd = ['ffmpeg', '-y', '-i', str(src), '-c:v', 'libx264', '-preset', 'fast', '-crf',
                   str(self.params.get('crf', 23)), str(tmp_video)]
            run_cmd(cmd)

        # 2) извлекаем аудио из оригинала
        self.status.emit('Извлечение аудио ...')
        extract_audio(src, tmp_audio)

        # 3) применяем аудио-фильтры через ffmpeg (highpass + fade-in/out)
        self.status.emit('Применение аудио-фильтров ...')
        audio_filters = []
        if params.get('audio_highpass', False):
            audio_filters.append(f"highpass=f={params.get('highpass_freq', 200)}")
        if params.get('audio_lowpass', False):
            audio_filters.append(f"lowpass=f={params.get('lowpass_freq', 12000)}")
        # fade-in/out: вычислим длительность и применим по концам
        dur = info['duration']
        fade_dur = params.get('audio_fade_dur', 0.8)
        if fade_dur > 0 and dur > 0:
            # создаём новый временный аудио файл с эффектами
            audio_tmp2 = out_dir / f"tmp_{src.stem}_afilter.aac"
            af = ','.join(audio_filters + [f"afade=t=in:st=0:d={fade_dur}",
                                           f"afade=t=out:st={max(0, dur - fade_dur)}:d={fade_dur}"])
            cmd_af = ['ffmpeg', '-y', '-i', str(tmp_audio), '-af', af, '-c:a', 'aac', '-b:a', '128k', str(audio_tmp2)]
            if run_cmd(cmd_af):
                tmp_audio = audio_tmp2

        # 4) финальная упаковка: перекодируем видео с нужными цветовыми фильтрами (если нужно) и встраиваем аудио
        self.status.emit('Финальная упаковка (цветокор и встраивание аудио) ...')
        # Можно добавить дополнительный ffmpeg-видеофильтр: hue/saturation/noise
        vf_filters = []
        # небольшая глобальная hue/saturation (для надежности)
        global_hue = params.get('global_hue_pct', 0.0)
        global_sat = params.get('global_saturation_mult', 1.0)
        if abs(global_hue) > 1e-6 or abs(global_sat - 1.0) > 1e-6:
            vf_filters.append(f"hue=h={global_hue}:s={global_sat}")
        # tiny grain using ffmpeg noise filter (if available)
        if params.get('global_noise', 0.0) > 1e-6:
            vf_filters.append(f"noise=alls={int(params.get('global_noise') * 10)}:allf=t")

        vf = ','.join(vf_filters) if vf_filters else None

        # built ffmpeg command for muxing
        cmd = ['ffmpeg', '-y', '-i', str(tmp_video), '-i', str(tmp_audio)]
        if vf:
            cmd += ['-vf', vf]
        cmd += ['-c:v', 'libx264', '-preset', 'fast', '-crf', str(self.params.get('crf', 23)), '-c:a', 'aac', '-b:a',
                '128k', str(final_video)]
        run_cmd(cmd)

        # Удалим временные аудио/видео
        try:
            if tmp_video.exists():
                tmp_video.unlink()
            if tmp_audio.exists():
                tmp_audio.unlink()
        except Exception:
            pass

        self.status.emit(f'Готово: {final_video.name}')
        return True

    def process_image(self, src: Path, out_dir: Path) -> bool:
        try:
            img = Image.open(src).convert('RGB')
        except Exception as e:
            self.status.emit(f'Ошибка открытия: {src.name} ({e})')
            return False

        params = self.params
        # базовые трансформации
        brightness = random.uniform(0.96, 1.04)
        contrast = random.uniform(0.96, 1.04)
        sharpness = random.uniform(0.95, 1.05)
        hue_shift = params.get('hue_shift_pct', 0.0)
        sat_mult = params.get('saturation_mult', 1.0)
        micro_rotate = params.get('micro_rotate_deg', 0.0)
        watermark_noise = params.get('watermark_noise', 0.0)

        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Sharpness(img).enhance(sharpness)

        # повернём чуть-чуть (micro_rotate) и вернём crop к исходному размеру
        if micro_rotate and abs(micro_rotate) > 1e-6:
            angle = random.uniform(-micro_rotate, micro_rotate)
            img = img.rotate(angle, resample=Image.BICUBIC, expand=False)

        # HSV tweak для Pillow: через numpy
        if abs(hue_shift) > 1e-6 or abs(sat_mult - 1.0) > 1e-6:
            arr = np.array(img).astype('float32') / 255.0
            # RGB -> HSV
            r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
            maxc = np.max(arr, axis=2)
            minc = np.min(arr, axis=2)
            v = maxc
            s = np.where(maxc == 0, 0, (maxc - minc) / maxc)
            # hue calc
            hue = np.zeros_like(maxc)
            mask = (maxc == r)
            hue[mask] = (g - mask * 0 - b[mask]) / (maxc[mask] - minc[mask] + 1e-6)
            # Не делаем сложных расчётов — применим простую коррекцию насыщенности
            s = np.clip(s * sat_mult, 0, 1)
            # обратно в RGB (упрощённо через v и s) — используем конвертацию через cv2 для надёжности
            img_cv = cv2.cvtColor((arr * 255).astype('uint8'), cv2.COLOR_RGB2HSV).astype('float32')
            img_cv[..., 1] = np.clip(img_cv[..., 1] * sat_mult, 0, 255)
            img_cv[..., 0] = (img_cv[..., 0] + hue_shift * 179) % 179
            img = Image.fromarray(cv2.cvtColor(img_cv.astype('uint8'), cv2.COLOR_HSV2RGB))

        # watermark noise
        if watermark_noise and watermark_noise > 1e-6:
            arr = np.array(img).astype('int32')
            noise = (np.random.randn(*arr.shape) * 255 * watermark_noise).astype('int32')
            arr = np.clip(arr + noise, 0, 255).astype('uint8')
            img = Image.fromarray(arr)

        out_name = f"enh_{src.stem}.jpg"
        out_path = out_dir / out_name
        quality = random.randint(88, 96)
        img.save(out_path, format='JPEG', quality=quality)
        self.status.emit(f'Готово: {out_name}')
        return True


class DragDropList(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self.addItem(path)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Уникализатор Lazz.org — Artist Video Enhancer')
        self.setMinimumSize(1000, 650)
        self.worker = None

        # Устанавливаем тёмную тему
        self.set_dark_theme()

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        left = QVBoxLayout()
        self.list_widget = DragDropList()
        left.addWidget(QLabel('Перетащите видео или изображения сюда'))
        left.addWidget(self.list_widget)

        btns = QHBoxLayout()
        add_btn = QPushButton('Добавить файлы')
        add_btn.clicked.connect(self.add_files)
        remove_btn = QPushButton('Удалить выбранные')
        remove_btn.clicked.connect(self.remove_selected)
        clear_btn = QPushButton('Очистить список')
        clear_btn.clicked.connect(self.list_widget.clear)
        btns.addWidget(add_btn);
        btns.addWidget(remove_btn);
        btns.addWidget(clear_btn)
        left.addLayout(btns)

        main_layout.addLayout(left, 3)

        right = QVBoxLayout()
        right.addWidget(QLabel('Настройки уникализации'))

        out_layout = QHBoxLayout()
        self.out_edit = QLineEdit(str(Path.cwd() / 'output'))
        out_btn = QPushButton('Обзор')
        out_btn.clicked.connect(self.select_output)
        out_layout.addWidget(QLabel('Папка вывода:'))
        out_layout.addWidget(self.out_edit)
        out_layout.addWidget(out_btn)
        right.addLayout(out_layout)

        # Preset
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel('Уровень уникализации:'))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(['Мягкий', 'Средний', 'Максимальный'])
        preset_layout.addWidget(self.preset_combo)
        right.addLayout(preset_layout)

        # Frame ops
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel('Вставка кадров (сек):'))
        self.insert_every_spin = QSpinBox();
        self.insert_every_spin.setRange(1, 60);
        self.insert_every_spin.setValue(5)
        frame_layout.addWidget(self.insert_every_spin)
        frame_layout.addWidget(QLabel('Кол-во кадров:'))
        self.insert_count_spin = QSpinBox();
        self.insert_count_spin.setRange(0, 5);
        self.insert_count_spin.setValue(1)
        frame_layout.addWidget(self.insert_count_spin)
        frame_layout.addWidget(QLabel('Удаление (%):'))
        self.remove_prob_spin = QSpinBox();
        self.remove_prob_spin.setRange(0, 50);
        self.remove_prob_spin.setValue(0)
        frame_layout.addWidget(self.remove_prob_spin)
        right.addLayout(frame_layout)

        # Micro-rotate / color
        ms_layout = QHBoxLayout()
        ms_layout.addWidget(QLabel('Микроповорот (°):'))
        self.micro_rotate_spin = QSpinBox();
        self.micro_rotate_spin.setRange(0, 5);
        self.micro_rotate_spin.setValue(0)
        ms_layout.addWidget(self.micro_rotate_spin)
        ms_layout.addWidget(QLabel('Оттенок (%):'))
        self.hue_spin = QSpinBox();
        self.hue_spin.setRange(-10, 10);
        self.hue_spin.setValue(0)
        ms_layout.addWidget(self.hue_spin)
        ms_layout.addWidget(QLabel('Насыщенность (%):'))
        self.sat_spin = QSpinBox();
        self.sat_spin.setRange(80, 120);
        self.sat_spin.setValue(100)
        ms_layout.addWidget(self.sat_spin)
        right.addLayout(ms_layout)

        # Noise and audio
        na_layout = QHBoxLayout()
        self.noise_spin = QSpinBox();
        self.noise_spin.setRange(0, 10);
        self.noise_spin.setValue(0)
        na_layout.addWidget(QLabel('Шум (%):'))
        na_layout.addWidget(self.noise_spin)
        self.audio_hp_cb = QCheckBox('Фильтр высоких частот')
        na_layout.addWidget(self.audio_hp_cb)
        na_layout.addWidget(QLabel('Плавность (сек):'))
        self.fade_spin = QSpinBox();
        self.fade_spin.setRange(0, 5);
        self.fade_spin.setValue(1)
        na_layout.addWidget(self.fade_spin)
        right.addLayout(na_layout)

        # CRF
        crf_layout = QHBoxLayout()
        crf_layout.addWidget(QLabel('Качество видео:'))
        self.crf_spin = QSpinBox();
        self.crf_spin.setRange(18, 32);
        self.crf_spin.setValue(23)
        crf_layout.addWidget(self.crf_spin)
        right.addLayout(crf_layout)

        # Кнопки
        self.start_btn = QPushButton('Начать уникализацию')
        self.start_btn.clicked.connect(self.start_process)
        self.cancel_btn = QPushButton('Отмена')
        self.cancel_btn.clicked.connect(self.cancel_process)
        self.cancel_btn.setEnabled(False)
        right.addWidget(self.start_btn)
        right.addWidget(self.cancel_btn)

        # Ссылки
        links_layout = QHBoxLayout()
        github_btn = QPushButton('GitHub')
        github_btn.clicked.connect(lambda: self.open_url('https://github.com/lazzteam'))
        telegram_btn = QPushButton('Telegram')
        telegram_btn.clicked.connect(lambda: self.open_url('https://t.me/lazzteam'))
        website_btn = QPushButton('Lazz.org')
        website_btn.clicked.connect(lambda: self.open_url('https://lazz.org'))
        links_layout.addWidget(github_btn)
        links_layout.addWidget(telegram_btn)
        links_layout.addWidget(website_btn)
        right.addLayout(links_layout)

        right.addStretch()
        self.status_label = QLabel('Готов к работе')
        self.progress = QProgressBar();
        self.progress.setValue(0)
        right.addWidget(self.status_label)
        right.addWidget(self.progress)

        main_layout.addLayout(right, 2)

    def set_dark_theme(self):
        """Устанавливает тёмную тему для приложения"""
        app = QApplication.instance()
        app.setStyle('Fusion')

        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, QColor(35, 35, 35))

        app.setPalette(dark_palette)
        app.setStyleSheet("""
            QMainWindow {
                background-color: #353535;
            }
            QPushButton {
                background-color: #555555;
                color: white;
                border: 1px solid #666666;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
            QPushButton:pressed {
                background-color: #777777;
            }
            QListWidget {
                background-color: #353535;
                color: white;
                border: 1px solid #666666;
            }
            QLineEdit {
                background-color: #454545;
                color: white;
                border: 1px solid #666666;
                padding: 2px;
            }
            QSpinBox {
                background-color: #454545;
                color: white;
                border: 1px solid #666666;
                padding: 2px;
            }
            QComboBox {
                background-color: #454545;
                color: white;
                border: 1px solid #666666;
                padding: 2px;
            }
            QCheckBox {
                color: white;
            }
            QLabel {
                color: white;
            }
            QProgressBar {
                border: 1px solid #666666;
                border-radius: 3px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #2a82da;
                width: 20px;
            }
        """)

    def open_url(self, url):
        """Открывает URL в браузере по умолчанию"""
        import webbrowser
        webbrowser.open(url)

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, 'Выберите файлы', str(Path.cwd()),
                                                'Media Files (*.mp4 *.mov *.avi *.mkv *.webm *.jpg *.jpeg *.png *.webp *.bmp)')
        for f in files:
            self.list_widget.addItem(f)

    def remove_selected(self):
        for item in self.list_widget.selectedItems():
            self.list_widget.takeItem(self.list_widget.row(item))

    def select_output(self):
        d = QFileDialog.getExistingDirectory(self, 'Папка вывода', str(Path.cwd()))
        if d:
            self.out_edit.setText(d)

    def start_process(self):
        if self.list_widget.count() == 0:
            QMessageBox.warning(self, 'Нет файлов', 'Добавьте файлы для обработки')
            return
        tasks = [self.list_widget.item(i).text() for i in range(self.list_widget.count())]

        # Собираем параметры
        preset = self.preset_combo.currentText()
        # базовые параметры на основе пресета
        if preset == 'Мягкий':
            preset_factors = {'insert_count': 1, 'remove_prob': 0.0, 'micro_rotate_deg': 0.1, 'hue_shift_pct': 0.0,
                              'saturation_mult': 1.02, 'watermark_noise': 0.002}
        elif preset == 'Средний':
            preset_factors = {'insert_count': 1, 'remove_prob': 0.02, 'micro_rotate_deg': 0.2, 'hue_shift_pct': 0.01,
                              'saturation_mult': 1.04, 'watermark_noise': 0.005}
        else:  # Максимальный
            preset_factors = {'insert_count': 2, 'remove_prob': 0.05, 'micro_rotate_deg': 0.4, 'hue_shift_pct': 0.02,
                              'saturation_mult': 1.06, 'watermark_noise': 0.01}

        params = {
            'output_dir': self.out_edit.text() or str(Path.cwd() / 'output'),
            'crf': self.crf_spin.value(),
            'insert_every': self.insert_every_spin.value(),
            'insert_count': self.insert_count_spin.value() or preset_factors['insert_count'],
            'remove_prob': (self.remove_prob_spin.value() / 100.0) or preset_factors['remove_prob'],
            'micro_rotate_deg': float(self.micro_rotate_spin.value()) or preset_factors['micro_rotate_deg'],
            'hue_shift_pct': (self.hue_spin.value() / 100.0) or preset_factors['hue_shift_pct'],
            'saturation_mult': (self.sat_spin.value() / 100.0) or preset_factors['saturation_mult'],
            'watermark_noise': (self.noise_spin.value() / 100.0) or preset_factors['watermark_noise'],
            'audio_highpass': self.audio_hp_cb.isChecked(),
            'highpass_freq': 200,
            'audio_fade_dur': float(self.fade_spin.value()),
            'global_hue_pct': 0.0,
            'global_saturation_mult': 1.0,
            'global_noise': 0.0
        }

        # Блокируем UI
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status_label.setText('Запуск уникализации...')
        self.progress.setValue(0)

        # Запуск воркера
        self.worker = Worker(tasks, params)
        self.worker.progress.connect(self.on_progress)
        self.worker.status.connect(self.on_status)
        self.worker.finished_all.connect(self.on_finished)
        self.worker.start()

    def cancel_process(self):
        if self.worker:
            self.worker.stop()
            self.status_label.setText('Отмена...')

    def on_progress(self, val):
        self.progress.setValue(val)

    def on_status(self, text):
        self.status_label.setText(text)

    def on_finished(self):
        self.status_label.setText('Уникализация завершена!')
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress.setValue(100)
        QMessageBox.information(self, 'Готово', 'Обработка завершена. Проверьте папку вывода.')


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
