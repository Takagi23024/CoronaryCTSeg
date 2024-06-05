import vtk
from vtk.util import numpy_support

# DICOMファイルが格納されているディレクトリのパスを設定
PathDicom = "./data/MIE034_CT/"

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


# %%


def detect_circles_in_slices(closedImage, ConstPixelDims, hough_radii):
    hough_res = []
    for i in range(ConstPixelDims[2]):
        slice_image = closedImage[:, :, i]
        edges = feature.canny(slice_image.astype(float))
        hough_res.append(transform.hough_circle(edges, hough_radii))

    accums, cx, cy, radii, z_slices = [], [], [], [], []
    for z, res in enumerate(hough_res):
        for radius, h in zip(hough_radii, res):
            peaks = transform.hough_circle_peaks([h], [radius], total_num_peaks=1)
            accums.extend(peaks[0])
            cx.extend(peaks[1])
            cy.extend(peaks[2])
            radii.extend([radius] * len(peaks[0]))
            z_slices.extend([z] * len(peaks[0]))

    return accums, cx, cy, radii, z_slices


# 円検出のための半径の範囲を設定
hough_radii = np.arange(20, 50, 2)
accums, cx, cy, radii, z_slices = detect_circles_in_slices(closedImage, ConstPixelDims, hough_radii)

# 検出された円に対応するピクセル部分に色を塗る
coloredArray = np.zeros_like(ArrayDicom)
for x, y, z, r in zip(cx, cy, z_slices, radii):
    rr, cc = np.ogrid[: ConstPixelDims[0], : ConstPixelDims[1]]
    mask = (rr - x) ** 2 + (cc - y) ** 2 <= r**2
    coloredArray[mask, z] = 255  # 色を塗る（ここでは255を使用）

# coloredArrayのポイントクラウドを作成
points = np.argwhere(coloredArray > 0)
colors = np.tile([1, 0, 0], (len(points), 1))  # 赤色

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# .pcdファイルに保存
o3d.io.write_point_cloud("colored_circles.pcd", pcd)


# %%
def detect_circles_in_slices(closedImage, ConstPixelDims, hough_radii):
    """
    Hough変換を用いてスライスごとに円を検出し、中心座標、半径、輝度値を取得する関数。

    Parameters:
    closedImage (numpy.ndarray): 3次元バイナリ画像
    ConstPixelDims (list): 画像の次元 [x, y, z]
    hough_radii (numpy.ndarray): 検出する円の半径の範囲

    Returns:
    accums (list): 円の累積投票数（強度）
    cx (list): 円の中心のx座標
    cy (list): 円の中心のy座標
    radii (list): 円の半径
    z_slices (list): 円のz座標（スライス番号）
    """
    hough_res = []
    for i in range(ConstPixelDims[2]):
        slice_image = closedImage[:, :, i]
        edges = feature.canny(slice_image.astype(float))
        hough_res.append(transform.hough_circle(edges, hough_radii))

    accums, cx, cy, radii, z_slices = [], [], [], [], []
    for z, res in enumerate(hough_res):
        for radius, h in zip(hough_radii, res):
            peaks = transform.hough_circle_peaks([h], [radius], total_num_peaks=1)
            accums.extend(peaks[0])
            cx.extend(peaks[1])
            cy.extend(peaks[2])
            radii.extend([radius] * len(peaks[0]))
            z_slices.extend([z] * len(peaks[0]))

    return accums, cx, cy, radii, z_slices


def create_circle_pcd(center_x, center_y, radius, z, num_points=100):
    """
    円のポイントクラウドを作成する関数。

    Parameters:
    center_x (int): 円の中心のx座標
    center_y (int): 円の中心のy座標
    radius (int): 円の半径
    z (int): 円のz座標（スライス番号）
    num_points (int): ポイントクラウドの点数

    Returns:
    pcd (open3d.geometry.PointCloud): 円のポイントクラウド
    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    z = np.full_like(x, z)

    points = np.column_stack((x, y, z)).astype(np.float64)
    colors = np.tile([0, 0, 1], (num_points, 1))  # blue

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


# 使用例
hough_radii = np.arange(20, 40, 2)
accums, cx, cy, radii, z_slices = detect_circles_in_slices(closedImage, ConstPixelDims, hough_radii)

# 全ての検出された円をポイントクラウドとして作成
all_circles_pcd = o3d.geometry.PointCloud()
for center_x, center_y, radius, z in zip(cx, cy, radii, z_slices):
    circle_pcd = create_circle_pcd(center_x, center_y, radius, z, num_points=100)
    all_circles_pcd += circle_pcd

# .pcdファイルに保存
o3d.io.write_point_cloud("all_circles.pcd", all_circles_pcd)

print("全ての検出された円のポイントクラウドをall_circles.pcdに保存しました。")
# %%

import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# シード点の設定
seed_x, seed_y, seed_z = 160, 247, 50

# 関心領域（ROI）の設定
roi_margin = 50  # シード点からの範囲
z_upper_limit = seed_z  # z方向の上限

# ROIの抽出
roi = ArrayDicom[
    max(0, seed_x - roi_margin) : min(ConstPixelDims[0], seed_x + roi_margin),
    max(0, seed_y - roi_margin) : min(ConstPixelDims[1], seed_y + roi_margin),
    max(0, seed_z - roi_margin) : z_upper_limit,
]

# K-meansクラスタリングの適用
num_clusters = 3  # 心臓、左冠動脈、右冠動脈の3クラスタ
X = roi.reshape(-1, 1)
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
labels = kmeans.labels_

# クラスタリング結果を元の形状にリシェイプ
clustered_roi = labels.reshape(roi.shape)

# クラスタごとのマスクを作成
heart_mask = clustered_roi == 0
left_coronary_mask = clustered_roi == 1
right_coronary_mask = clustered_roi == 2

# 3D描画の準備
x, y, z = np.indices(clustered_roi.shape)

fig = go.Figure()

# Heart
fig.add_trace(
    go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=heart_mask.flatten(),
        isomin=0.5,
        isomax=1.5,
        surface_count=1,
        colorscale="Reds",
        showscale=False,
        name="Heart",
        visible=True,
    )
)

# Left Coronary Artery
fig.add_trace(
    go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=left_coronary_mask.flatten(),
        isomin=0.5,
        isomax=1.5,
        surface_count=1,
        colorscale="Blues",
        showscale=False,
        name="Left Coronary Artery",
        visible=False,
    )
)

# Right Coronary Artery
fig.add_trace(
    go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=right_coronary_mask.flatten(),
        isomin=0.5,
        isomax=1.5,
        surface_count=1,
        colorscale="Greens",
        showscale=False,
        name="Right Coronary Artery",
        visible=False,
    )
)

# レイアウト設定
fig.update_layout(
    scene=dict(xaxis=dict(title="X"), yaxis=dict(title="Y"), zaxis=dict(title="Z")),
    updatemenus=[
        dict(
            buttons=list(
                [
                    dict(label="Heart", method="update", args=[{"visible": [True, False, False]}, {"title": "Heart"}]),
                    dict(
                        label="Left Coronary Artery",
                        method="update",
                        args=[{"visible": [False, True, False]}, {"title": "Left Coronary Artery"}],
                    ),
                    dict(
                        label="Right Coronary Artery",
                        method="update",
                        args=[{"visible": [False, False, True]}, {"title": "Right Coronary Artery"}],
                    ),
                    dict(label="All", method="update", args=[{"visible": [True, True, True]}, {"title": "All"}]),
                ]
            ),
            direction="down",
            showactive=True,
        ),
    ],
)

fig.show()
