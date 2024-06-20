import numpy as np
import pydicom


def read_angio_dicom(angio_dicom_path):
    """
    血管造影DICOMファイルを読み込み、関連情報を抽出する。

    パラメータ:
    angio_dicom_path (str): DICOMファイルのパス。

    戻り値:
    tuple: 以下の要素を含むタプル:
        - ArrayDicom (numpy.ndarray): DICOMピクセル配列 (幅, 高さ, スライス)。
        - total_slices (int): 総スライス数。
        - distance_source_to_patient (float): カメラと患者の距離。
        - horizontal_degree (float): 水平角度 ("+" -> LAO側, "-" -> RAO側)。
        - vertical_degree (float): 垂直角度 ("+" -> CRA側, "-" -> CAU側)。
    """
    # DICOMファイルを読み込む
    dicom_data = pydicom.dcmread(angio_dicom_path)

    # ピクセル配列を抽出
    ArrayDicom = dicom_data.pixel_array  # (スライス, 高さ, 幅)
    ArrayDicom = np.transpose(ArrayDicom, (2, 1, 0))  # -> (幅, 高さ, スライス)
    ArrayDicom = ArrayDicom[:, ::-1, :]  # 配列を上下反転
    total_slices = ArrayDicom.shape[2]  # 総スライス数を取得

    # カメラと患者の距離を抽出
    distance_source_to_patient = float(dicom_data[(0x0018, 0x1111)].value)

    # アーム角度を抽出
    horizontal_degree = float(dicom_data[(0x0018, 0x1510)].value)
    vertical_degree = float(dicom_data[(0x0018, 0x1511)].value)

    return ArrayDicom, total_slices, distance_source_to_patient, horizontal_degree, vertical_degree


def calculate_rotation_angles(axis_vector, center_point, target_point):
    """
    主軸ベクトル、中心点、対象点から水平角度と垂直角度を計算する。

    パラメータ:
    axis_vector (tuple): 主軸ベクトルの座標 (x, y, z)。
    center_point (tuple): 中心点の座標 (x, y, z)。
    target_point (tuple): 対象点の座標 (x, y, z)。

    戻り値:
    tuple: 水平角度 (horizontal_degree) と垂直角度 (vertical_degree)。
    """
    # ベクトルを計算
    vector = np.array(target_point) - np.array(center_point)
    axis_vector = np.array(axis_vector)

    # 水平角度 (XY平面に投影したときの角度)
    horizontal_degree = np.degrees(np.arctan2(vector[1], vector[0]) - np.arctan2(axis_vector[1], axis_vector[0]))

    # 垂直角度 (XZ平面に投影したときの角度)
    vertical_degree = np.degrees(
        np.arctan2(vector[2], np.sqrt(vector[0] ** 2 + vector[1] ** 2))
        - np.arctan2(axis_vector[2], np.sqrt(axis_vector[0] ** 2 + axis_vector[1] ** 2))
    )

    return horizontal_degree, vertical_degree


def rotation_matrix_from_axis_angle(axis, theta):
    """
    回転軸と回転角度から回転行列を生成する。

    パラメータ:
    axis (numpy.ndarray): 回転軸ベクトル。
    theta (float): 回転角度（ラジアン）。

    戻り値:
    numpy.ndarray: 3x3の回転行列。
    """
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def combined_rotation_matrix(axis, horizontal_degree, vertical_degree):
    """
    主軸ベクトル、水平角度、垂直角度から回転行列を生成する。

    パラメータ:
    axis (tuple): 主軸ベクトル (x, y, z)。
    horizontal_degree (float): 水平角度（度）。
    vertical_degree (float): 垂直角度（度）。

    戻り値:
    numpy.ndarray: 3x3の回転行列。
    """
    # 度をラジアンに変換
    horizontal_radian = np.radians(horizontal_degree)
    vertical_radian = np.radians(vertical_degree)

    # 主軸ベクトルを正規化
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)

    # 水平回転行列を生成
    horizontal_rotation_matrix = rotation_matrix_from_axis_angle(axis, horizontal_radian)

    # 垂直回転行列を生成
    vertical_rotation_matrix = rotation_matrix_from_axis_angle(axis, vertical_radian)

    # 回転行列を組み合わせる
    combined_matrix = np.dot(horizontal_rotation_matrix, vertical_rotation_matrix)

    return combined_matrix


if __name__ == "__main__":
    axis_vector = (0, 1, 0)
    center_point = (0, 0, 0)
