import os
import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img

SUB_ID = "sub-p019"

BASE_PATH = "./ds004873"
OUTPUT_DIR = "./labels"
os.makedirs(OUTPUT_DIR, exist_ok=True)

Z_THRESH = 2.3               # порог значимости BOLD сигнала Z для активации
# данное значение соответствует примерно 0.01 p-value
CMRO2_MIN_ABS_CHANGE = 0.01   # минимальное значимое абсолютное изменение CMRO2 (в единицах карты)

# Какие файлы используются:
# 1. Структурная анатомия anat_..._T1w.nii.gz
# 2. BOLD контраст из деривативов/func/1stlevel_calccontol_space_T2
# Внутри z-статистика, показывающая насколько значимо изменился болд сигнал
# 3. Количественные карты CBF: task-calc_space-T1w и для control
# Внутри абсолютные значения мозгового кровотока
# 4. Количественные карты CMRO2 в task-calc_cpace-T1w_desc-orig_cmro2
# аналогично для control - количественная карта CMRO2




# ----------------------------------------------------------------------
# 1. ЗАГРУЗКА ДАННЫХ
# ----------------------------------------------------------------------
print(f"Обработка {SUB_ID}...")

# T1w
t1_path = os.path.join(BASE_PATH, SUB_ID, "anat", f"{SUB_ID}_T1w.nii.gz")
t1_img = nib.load(t1_path)
t1_data = t1_img.get_fdata().astype(np.float32) # получили трехмерный массив чисел - интенсивность вокселей
print(f"  T1w: {t1_data.shape}")

# BOLD контраст (CALC vs CONTROL) в T2-пространстве
contrast_path = os.path.join(
    BASE_PATH, "derivatives", SUB_ID, "func",
    f"{SUB_ID}_1stlevel_calccontrol_space-T2.nii.gz"
)
contrast_img = nib.load(contrast_path)
contrast_data = contrast_img.get_fdata().astype(np.float32)  # Загрузили карту контраста CALC - CONTROL
print(f"  BOLD контраст: {contrast_data.shape}")

# Количественные карты в T1w-пространстве (используем оригинальные)
calc_cbf_path = os.path.join(BASE_PATH, "derivatives", SUB_ID, "qmri",
                             f"{SUB_ID}_task-calc_space-T1w_cbf.nii.gz")
ctrl_cbf_path = os.path.join(BASE_PATH, "derivatives", SUB_ID, "qmri",
                             f"{SUB_ID}_task-control_space-T1w_cbf.nii.gz")

calc_cmro2_path = os.path.join(BASE_PATH, "derivatives", SUB_ID, "qmri",
                               f"{SUB_ID}_task-calc_space-T1w_desc-orig_cmro2.nii.gz")
ctrl_cmro2_path = os.path.join(BASE_PATH, "derivatives", SUB_ID, "qmri",
                               f"{SUB_ID}_task-control_space-T1w_desc-orig_cmro2.nii.gz")

# Количественные карты мозгового кровотока CBF и метаболизма кислорода CMRO2 для CALC И CONTROL В T1
calc_cbf = nib.load(calc_cbf_path).get_fdata().astype(np.float32)
ctrl_cbf = nib.load(ctrl_cbf_path).get_fdata().astype(np.float32)
calc_cmro2 = nib.load(calc_cmro2_path).get_fdata().astype(np.float32)
ctrl_cmro2 = nib.load(ctrl_cmro2_path).get_fdata().astype(np.float32)
print("  Количественные карты загружены.")

# ----------------------------------------------------------------------
# 2. ПРИВЕДЕНИЕ ФОРМ
# ----------------------------------------------------------------------
if calc_cbf.shape != t1_data.shape:
    calc_cbf = np.transpose(calc_cbf, (1, 2, 0))
    ctrl_cbf = np.transpose(ctrl_cbf, (1, 2, 0))
    calc_cmro2 = np.transpose(calc_cmro2, (1, 2, 0))
    ctrl_cmro2 = np.transpose(ctrl_cmro2, (1, 2, 0))
# Если шейп не совпал, переставляем, чтобы сопоставлять воксели

# ----------------------------------------------------------------------
# 3. РЕСЕМПЛИНГ BOLD В ПРОСТРАНСТВО T1w
# ----------------------------------------------------------------------
print("  Ресемплинг BOLD в T1w...")

#Переводим BOLD в пространство T1w
contrast_resampled_img = resample_to_img(contrast_img, t1_img, interpolation='linear')
contrast_t1w = contrast_resampled_img.get_fdata().astype(np.float32)
out_contrast = os.path.join(OUTPUT_DIR, f"{SUB_ID}_calccontrol_space-T1w.nii.gz")
nib.save(contrast_resampled_img, out_contrast)
# Сохраняем пересчитанный контраст для
# визуального контроля и использования в дальнейшем
print(f"  Сохранён: {out_contrast}")

# ----------------------------------------------------------------------
# 4. АБСОЛЮТНЫЕ ИЗМЕНЕНИЯ
# ----------------------------------------------------------------------
delta_cmro2_abs = calc_cmro2 - ctrl_cmro2 # считаем абсолютное изменение cmro2
# (CBF нам не нужен для упрощённого правила, но оставим для информации)

# ----------------------------------------------------------------------
# 5. МАСКА МОЗГА И МАСКА АКТИВНЫХ ВОКСЕЛЕЙ
# ----------------------------------------------------------------------

brain_mask_path = os.path.join( BASE_PATH, "derivatives", SUB_ID, "anat", f"{SUB_ID}_desc-fmriprep_brain_mask.nii.gz")
brain_mask_img = nib.load(brain_mask_path)
brain_mask = brain_mask_img.get_fdata().astype(bool)

# Если форма маски не совпадает с T1w, ресемплируем её к T1w
if brain_mask.shape != t1_data.shape:
    print("  Ресемплинг маски мозга к T1w...")
    brain_mask = resample_to_img(brain_mask_img, t1_img, interpolation='nearest').get_fdata().astype(bool)
else:
    print("  Маска мозга загружена, форма совпадает.")

mask_active = (np.abs(contrast_t1w) > Z_THRESH) & brain_mask # отбросим незначимые слабоактивные воксели, чтобы получить только достоверно определенные
print(f"  Активных вокселей: {np.sum(mask_active)}")

# ----------------------------------------------------------------------
# 6. КЛАССИФИКАЦИЯ (УПРОЩЁННОЕ ПРАВИЛО)
# ----------------------------------------------------------------------
label_map = np.zeros_like(t1_data, dtype=np.uint8) # итоговая карта разметки
num_concordant = 0
num_discordant = 0

# проходим циклом по активным вокселям
for idx in zip(*np.where(mask_active)):
    # извлечем значения Z-статистики и абсолютного изменения CMRO2
    bold_val = contrast_t1w[idx]
    cmro2_abs = delta_cmro2_abs[idx]

    # пропукаем воксели с некорректными значениями и малым cmro2
    if np.isnan(cmro2_abs) or np.isinf(cmro2_abs):
        continue
    if abs(cmro2_abs) < CMRO2_MIN_ABS_CHANGE:
        continue
    # Упрощенная Логика: если знаки изменения BOLD-сигнала и CMRO2 совпали,
    # значит у нас конкордантный тип, иначе - дискордантный
    if np.sign(bold_val) == np.sign(cmro2_abs):
        label_map[idx] = 1
        num_concordant += 1
    else:
        label_map[idx] = 2
        num_discordant += 1

# Статистика
total = num_concordant + num_discordant
print(f"  Конкордантных: {num_concordant}")
print(f"  Дискордантных: {num_discordant}")
print(f" Вего прошли фильтрацию: {num_discordant + num_concordant}")
if total > 0:
    print(f"  Доля дискордантных: {num_discordant / total * 100:.1f}%")

# ----------------------------------------------------------------------
# 7. СОХРАНЕНИЕ РАЗМЕТКИ
# ----------------------------------------------------------------------
label_img = nib.Nifti1Image(label_map, affine=t1_img.affine, header=t1_img.header)
label_path = os.path.join(OUTPUT_DIR, f"{SUB_ID}_label-simple.nii.gz")
nib.save(label_img, label_path)
print(f"  Разметка сохранена: {label_path}")
