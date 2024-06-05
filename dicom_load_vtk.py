# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import plotly.express as px
import scipy.ndimage as ndimage
import vtk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import draw, feature, filters, measure, morphology, transform
from skimage.draw import disk
from vtk.util import numpy_support

# DICOMファイルが格納されているディレクトリのパスを設定
PathDicom = "./data/MIE022_CT/"

# vtkDICOMImageReaderオブジェクトを作成し、DICOMファイルのディレクトリを設定してデータを読み込む
reader = vtk.vtkDICOMImageReader()
reader.SetDirectoryName(PathDicom)
reader.Update()

# 画像の次元を取得
_extent = reader.GetDataExtent()
ConstPixelDims = [
    _extent[1] - _extent[0] + 1,
    _extent[3] - _extent[2] + 1,
    _extent[5] - _extent[4] + 1,
]

# ピクセル間の間隔（スライス間の距離）を取得
ConstPixelSpacing = reader.GetPixelSpacing()

# Rescale SlopeとRescale Interceptを取得
rescale_slope = reader.GetRescaleSlope() if reader.GetRescaleSlope() != 0 else 1
rescale_intercept = reader.GetRescaleOffset()

# vtkImageDataオブジェクトをreaderから取得
imageData = reader.GetOutput()
# vtkImageDataオブジェクトからvtkPointDataオブジェクトを取得
pointData = imageData.GetPointData()
# vtkPointDataオブジェクトに1つの配列しか含まれていないことを確認
assert pointData.GetNumberOfArrays() == 1
# numpy_support.vtk_to_numpy関数に必要なvtkArray（またはその派生型）を取得
arrayData = pointData.GetArray(0)

# vtkArrayをNumPy配列に変換
ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
# NumPy配列を3次元にリシェイプ（ConstPixelDimsをshapeとして使用）
ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order="F")

print(PathDicom)

print(f"ArrayDicom: {ArrayDicom[0]}")

# Hounsfield Unitsに変換
ArrayDicom = ArrayDicom * rescale_slope + rescale_intercept
print("ArrayDicom = ArrayDicom * rescale_slope + rescale_intercept")

print(f"rescale_intercept: {rescale_intercept}")
print(f"rescale_slope: {rescale_slope}")
print(f"ArrayDicom: {ArrayDicom[0]}")


def visualize_slices(ArrayDicom, start_slice=0, end_slice=None, slice_step=1, animation_speed=100):
    """
    指定された設定に基づいてスライスを可視化する関数。

    Parameters:
    - ArrayDicom: 3D NumPy配列
    - start_slice: 表示したいスライスの開始位置
    - end_slice: 表示したいスライスの終了位置
    - slice_step: 何枚ずつのスライスで表示するか
    - animation_speed: アニメーションの速度（ミリ秒単位）
    """
    # 転置して描画用に形状を変更
    img3d = np.transpose(ArrayDicom, (2, 1, 0))

    # end_sliceがNoneの場合、配列の最後のスライスを設定
    if end_slice is None:
        end_slice = img3d.shape[0]

    # スライスを可視化
    fig = px.imshow(
        img3d[start_slice:end_slice:slice_step, :, :],
        animation_frame=0,
        labels=dict(animation_frame="slice"),
        color_continuous_scale="jet",
        origin="lower",
    )

    # スライス番号を設定
    fig.update_layout(
        sliders=[
            {
                "steps": [
                    {
                        "args": [[f"{i}"]],
                        "label": f"{i}",
                        "method": "animate",
                    }
                    for i in range(start_slice, end_slice, slice_step)
                ],
                "currentvalue": {"prefix": "Slice: ", "font": {"size": 20}},
            }
        ],
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": animation_speed, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
    )

    fig.show()


def visualize_histgram(ArrayDicom, thresholds):

    # ヒストグラムをプロット
    plt.figure(figsize=(10, 6))
    plt.hist(
        ArrayDicom.flatten(), bins=range(int(np.min(ArrayDicom)), int(np.max(ArrayDicom)) + 100, 100), edgecolor="black"
    )
    plt.title("Distribution of ArrayDicom Values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)

    # 大津閾値をプロット
    for threshold in thresholds:
        plt.axvline(threshold, color="r", linestyle="dashed", linewidth=2)
        plt.text(threshold + 10, plt.ylim()[1] * 0.9, f"Otsu:{threshold}", color="r")

    # グラフ表示
    plt.show()


# visualize_slices(ArrayDicom, start_slice=0, end_slice=50, slice_step=1, animation_speed=100)

thresholds = filters.threshold_multiotsu(ArrayDicom)
print(thresholds)


filtered_array = ArrayDicom.copy()
filtered_array[ArrayDicom < thresholds[1]] = 0

visualize_slices(ArrayDicom=filtered_array, start_slice=0, end_slice=100, slice_step=5, animation_speed=100)


# %%

# クロージング処理で肺動脈部分を削除
start_step = time.time()
structure = np.ones((3, 3, 3), dtype=int)
# クロージング処理は、膨張（dilation）と収縮（erosion）を順に行うモルフォロジー操作
# 膨張（dilation）: バイナリ画像の白い領域（1の部分）を構造要素の形状に従って拡張します。
# 収縮（erosion）: 膨張後の画像に対して、構造要素の形状に従って白い領域を縮小します。
# 小さな黒いノイズが除去され、白い領域の小さな穴が埋められる
closedImage = ndimage.binary_closing(binaryImage, structure=structure)
print(f"クロージング処理時間: {time.time() - start_step:.2f}秒")


# %%


# %%


# %%
# 上行大動脈から領域拡張法を実行
expanded_aorta = morphology.binary_dilation(aorta_mask, structure)

# 新しいシード点を設定し、再度領域拡張を行い、冠状動脈を抽出
# 冠状動脈が心臓を覆う性質を利用
seed_points = np.argwhere(expanded_aorta)
new_seed = seed_points[np.argmax(seed_points[:, 2])]

# 冠状動脈抽出のための領域拡張
coronary_mask = morphology.binary_dilation(expanded_aorta, structure)

# マスク画像を用いて不必要な領域への拡張を防止
mask_image = morphology.binary_erosion(coronary_mask, structure)
final_coronary_mask = coronary_mask & ~mask_image
print(f"冠状動脈の抽出時間: {time.time() - start_step:.2f}秒")

# 冠状動脈抽出後の画像表示
plt.figure(figsize=(10, 10))
plt.imshow(final_coronary_mask[:, :, ConstPixelDims[2] // 2], cmap="gray")
plt.title("冠状動脈抽出後")
plt.show()

# 2.5. 冠状動脈末端部の抽出
start_step = time.time()
# 冠状動脈の進行方向を計算し、その情報を用いて領域拡張の精度を向上
# 方向情報を考慮した領域拡張
# 細線化処理
skeleton = morphology.skeletonize_3d(final_coronary_mask)


def find_skeleton_endpoints(skel):
    endpoints = []
    for i in range(1, skel.shape[0] - 1):
        for j in range(1, skel.shape[1] - 1):
            for k in range(1, skel.shape[2] - 1):
                if skel[i, j, k]:
                    neighborhood = skel[i - 1 : i + 2, j - 1 : j + 2, k - 1 : k + 2]
                    if np.sum(neighborhood) == 2:  # One for the center and one for the endpoint
                        endpoints.append((i, j, k))
    return endpoints


# 端点の検出
endpoints = find_skeleton_endpoints(skeleton)

# 進行方向の計算と領域拡張の調整
for endpoint in endpoints:
    direction_vector = endpoint - new_seed
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    # 方向情報を用いた領域拡張（例：進行方向に沿ってしきい値を調整）
print(f"冠状動脈末端部の抽出時間: {time.time() - start_step:.2f}秒")

# 冠状動脈末端部抽出後の画像表示
plt.figure(figsize=(10, 10))
plt.imshow(skeleton[:, :, ConstPixelDims[2] // 2], cmap="gray")
plt.title("冠状動脈末端部抽出後")
plt.show()


# 3D可視化のための設定
def plot_3d(image, threshold=-300):
    p = image.transpose(2, 1, 0)
    verts, faces, _, _ = measure.marching_cubes(p, threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


# 3Dプロット
start_step = time.time()
plot_3d(final_coronary_mask, 0.5)
print(f"3Dプロット時間: {time.time() - start_step:.2f}秒")
