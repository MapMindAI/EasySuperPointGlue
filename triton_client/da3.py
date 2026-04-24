import tritonclient.grpc as grpcclient
import numpy as np
import cv2
import time
from superpoint import find_triton_model
import tritonclient.grpc as grpcclient
import numpy as np
import cv2
import time


class DepthAnything3:
    def __init__(
        self,
        triton_url,
        model_key="depthanything3",
        model_version="1",
        input_height=280,
        input_width=504,
        use_imagenet_norm=False,
        near_percentile=2,
        far_percentile=98,
        gamma=0.7,
        invert_vis=True,
        expected_num_images=None,  # None means do not enforce
        input_name="image",
    ):
        self.grpc_client = grpcclient.InferenceServerClient(url=triton_url, verbose=False)
        self.model_version = model_version
        self.model_name = find_triton_model(self.grpc_client, model_key)

        if self.model_name is None:
            raise ValueError(f"Cannot find Triton model containing key: {model_key}")

        self.input_height = input_height
        self.input_width = input_width
        self.use_imagenet_norm = use_imagenet_norm
        self.near_percentile = near_percentile
        self.far_percentile = far_percentile
        self.gamma = gamma
        self.invert_vis = invert_vis
        self.expected_num_images = expected_num_images
        self.input_name = input_name

        print(f"Start {self.model_name} from {triton_url}")

        self.desired_outputs = [
            grpcclient.InferRequestedOutput("depth"),
            grpcclient.InferRequestedOutput("intrinsics"),
            grpcclient.InferRequestedOutput("extrinsics"),
        ]

    def _preprocess_single_image(self, image_numpy):
        if image_numpy is None:
            raise ValueError("Input image is None")

        if image_numpy.ndim == 2:
            image_rgb = cv2.cvtColor(image_numpy, cv2.COLOR_GRAY2RGB)
        elif image_numpy.ndim == 3 and image_numpy.shape[2] == 3:
            # OpenCV input is usually BGR
            image_rgb = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported image shape: {image_numpy.shape}")

        orig_h, orig_w = image_rgb.shape[:2]

        image_resized = cv2.resize(
            image_rgb,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )

        x = image_resized.astype(np.float32) / 255.0

        if self.use_imagenet_norm:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            x = (x - mean) / std

        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        return image_rgb, (orig_h, orig_w), x

    def _depth_to_vis(self, depth):
        depth = depth.astype(np.float32)
        valid = np.isfinite(depth)
        if not np.any(valid):
            return np.zeros_like(depth, dtype=np.float32)

        v = depth[valid]
        dmin = np.percentile(v, self.near_percentile)
        dmax = np.percentile(v, self.far_percentile)

        depth = np.clip(depth, dmin, dmax)
        depth = (depth - dmin) / (dmax - dmin + 1e-8)

        if self.invert_vis:
            depth = 1.0 - depth

        depth = np.power(depth, self.gamma)
        return depth

    def _build_input_tensor(self, images):
        if not isinstance(images, (list, tuple)):
            raise TypeError("images must be a list or tuple of numpy images")
        if len(images) == 0:
            raise ValueError("images must contain at least one image")
        if self.expected_num_images is not None and len(images) != self.expected_num_images:
            raise ValueError(
                f"Expected {self.expected_num_images} images, got {len(images)}"
            )

        image_rgbs = []
        orig_sizes = []
        tensors = []

        for img in images:
            img_rgb, orig_size, x = self._preprocess_single_image(img)
            image_rgbs.append(img_rgb)
            orig_sizes.append(orig_size)
            tensors.append(x)

        # [N, 3, H, W]
        x = np.stack(tensors, axis=0)

        # [1, N, 3, H, W]
        x = np.expand_dims(x, axis=0).astype(np.float32)

        meta = {
            "image_rgbs": image_rgbs,
            "orig_sizes": orig_sizes,
            "num_images": len(images),
        }
        return x, meta

    def get_response(self, response, key):
        result = response.as_numpy(key)
        if result is None:
            raise RuntimeError(f"Triton returned no '{key}' output")
        return result

    def run(self, images):
        x, meta = self._build_input_tensor(images)

        inputs = []
        input_tensor = grpcclient.InferInput(self.input_name, x.shape, "FP32")
        input_tensor.set_data_from_numpy(x)
        inputs.append(input_tensor)

        response = self.grpc_client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=inputs,
            outputs=self.desired_outputs,
        )

        depth = self.get_response(response, "depth")
        extrinsics = self.get_response(response, "extrinsics")
        intrinsics = self.get_response(response, "intrinsics")

        # expected [1, N, H, W]
        if depth.ndim != 4 or depth.shape[0] != 1:
            raise ValueError(f"Unexpected depth output shape: {depth.shape}")

        num_images = meta["num_images"]
        if depth.shape[1] != num_images:
            raise ValueError(
                f"Output image count mismatch: input has {num_images}, output has {depth.shape[1]}"
            )

        depth_list = []
        intrinsics_list = []
        extrinsics_list = []

        for i in range(num_images):
            depth_i = depth[0, i]
            orig_h, orig_w = meta["orig_sizes"][i]
            depth_i_resized = cv2.resize(
                depth_i, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC
            )
            depth_list.append(depth_i_resized)
            intrinsics_list.append(intrinsics[0, i])
            extrinsics_list.append(extrinsics[0, i])


        return {
            "depth_list": depth_list,                   # list of HxW
            "intrinsics_list": intrinsics_list,           # list of 3x3
            "extrinsics_list": extrinsics_list,           # list of 3x4
        }

    def run_paths(self, image_paths):
        if not isinstance(image_paths, (list, tuple)):
            raise TypeError("image_paths must be a list or tuple")

        images = []
        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {p}")
            images.append(img)

        return self.run(images)

    def save_visualizations(self, result, prefix="depth"):
        for i, depth_i in enumerate(result["depth_list"]):
            depth_i_vis = self._depth_to_vis(depth_i)
            save_path = f"{prefix}_{i}.png"
            depth_u8 = (np.clip(depth_i_vis, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(save_path, depth_u8)
            print(f"Saved {save_path}")


if __name__ == "__main__":
    da3 = DepthAnything3(
        triton_url="0.0.0.0:8001",
        expected_num_images=None,
    )

    time_ms_begin = time.time() * 1000
    result = da3.run_paths([
        "data/00221.jpg",
        "data/00261.jpg",
        "data/00321.jpg",
        "data/00361.jpg",
        "data/00421.jpg",
        "data/00461.jpg",
        "data/00521.jpg",
        "data/00561.jpg",
        "data/00621.jpg",
        "data/00661.jpg",
    ])
    time_ms_end = time.time() * 1000
    print(f"{len(result["depth_list"])}, used {time_ms_end - time_ms_begin}ms")

    da3.save_visualizations(result, prefix="data/depth")
