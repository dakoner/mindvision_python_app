import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot, QMutex
from PySide6.QtGui import QImage
from utils import QMutexLocker

def precompute_ssim_constants(img):
    img = img.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu_sq = mu ** 2
    sigma_sq = cv2.filter2D(img ** 2, -1, window)[5:-5, 5:-5] - mu_sq
    
    return {
        "img": img,
        "mu": mu,
        "mu_sq": mu_sq,
        "sigma_sq": sigma_sq,
        "window": window
    }

def compute_ssim_cached(img1, ref_stats):
    C1 = 6.5025
    C2 = 58.5225
    
    img1 = img1.astype(np.float64)
    window = ref_stats["window"]
    
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    
    mu2 = ref_stats["mu"]
    mu2_sq = ref_stats["mu_sq"]
    sigma2_sq = ref_stats["sigma_sq"]
    img2 = ref_stats["img"]
    
    mu1_mu2 = mu1 * mu2
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def compute_ssim(img1, img2):
    # Legacy wrapper for non-cached usage if needed, or we can just compute on the fly
    stats = precompute_ssim_constants(img2)
    return compute_ssim_cached(img1, stats)

class MatchingWorker(QObject):
    result_ready = Signal(QImage)
    log_signal = Signal(str)
    qr_found_signal = Signal(str)
    ssim_score_signal = Signal(float)

    def __init__(self):
        super().__init__()
        self.detector = None
        self.bf = None
        self.template_img = None
        self.template_kp = None
        self.template_des = None
        self.is_matching_enabled = False
        self.mutex = QMutex()

        # ArUco
        self.aruco_dict = None
        self.aruco_params = None
        self.aruco_obj = None
        self.aruco_display = {"ids": True, "rejected": False}
        self.contour_params = {
            "mode": "Canny",
            "thresh_min": 50,
            "thresh_max": 150,
            "threshold": 127,
            "min_area": 100,
            "max_area": 100000,
            "fill": False,
            "box": False,
        }
        self.current_algo = "ORB"
        self.is_contours_enabled = False

        # QR Code
        self.qr_detector = cv2.QRCodeDetector()
        
        # SSIM
        self.ssim_ref_img = None
        self.ssim_cache = None
        
        self.last_params = {}

    @Slot(dict)
    def update_params(self, params):
        # Update detector based on params
        with QMutexLocker(self.mutex):
            # Check if params actually changed
            if params == self.last_params:
                return
            self.last_params = params.copy()

            try:
                # If 'algo' is provided, we are updating the Matching Algo
                if "algo" in params:
                    self.current_algo = params["algo"]

                    if self.current_algo == "ORB":
                        self.detector = cv2.ORB_create(
                            nfeatures=params.get("nfeatures", 500),
                            scaleFactor=params.get("scaleFactor", 1.2),
                            nlevels=params.get("nlevels", 8),
                            edgeThreshold=params.get("edgeThreshold", 31),
                            firstLevel=params.get("firstLevel", 0),
                            WTA_K=params.get("WTA_K", 2),
                            scoreType=params.get("scoreType", 0),
                            patchSize=params.get("patchSize", 31),
                            fastThreshold=params.get("fastThreshold", 20),
                        )
                        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    elif self.current_algo == "SIFT":
                        self.detector = cv2.SIFT_create(
                            nfeatures=params.get("nfeatures", 0),
                            nOctaveLayers=params.get("nOctaveLayers", 3),
                            contrastThreshold=params.get("contrastThreshold", 0.04),
                            edgeThreshold=params.get("edgeThreshold", 10),
                            sigma=params.get("sigma", 1.6),
                        )
                        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                    elif self.current_algo == "AKAZE":
                        self.detector = cv2.AKAZE_create(
                            descriptor_type=params.get("descriptor_type", 5),
                            threshold=params.get("threshold", 0.0012),
                            nOctaves=params.get("nOctaves", 4),
                            nOctaveLayers=params.get("nOctaveLayers", 4),
                        )
                        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    elif self.current_algo == "ARUCO":
                        dict_name = params.get("dict", "DICT_4X4_50")
                        if hasattr(cv2.aruco, dict_name):
                            self.aruco_dict = cv2.aruco.getPredefinedDictionary(
                                getattr(cv2.aruco, dict_name)
                            )
                            if self.aruco_params is None:
                                self.aruco_params = cv2.aruco.DetectorParameters()

                            # Update ArUco Params
                            if "markerBorderBits" in params:
                                self.aruco_params.markerBorderBits = params[
                                    "markerBorderBits"
                                ]

                            self.aruco_display["ids"] = params.get("show_ids", True)
                            self.aruco_display["rejected"] = params.get(
                                "show_rejected", False
                            )

                            # Create ArucoDetector
                            try:
                                self.aruco_obj = cv2.aruco.ArucoDetector(
                                    self.aruco_dict, self.aruco_params
                                )
                            except AttributeError:
                                self.aruco_obj = None
                                self.log_signal.emit(
                                    "Error: cv2.aruco.ArucoDetector not found"
                                )
                        else:
                            self.log_signal.emit(f"Unknown ArUco dict: {dict_name}")
                            self.aruco_dict = None
                            self.aruco_obj = None
                    elif self.current_algo == "QRCODE":
                        # QR Code detector is already initialized
                        pass
                    elif self.current_algo == "HOUGH_CIRCLE":
                        # Hough Circle does not require a detector object, just params
                        pass

                    # Recompute template if exists (feature matching)
                    if (
                        self.current_algo in ["ORB", "SIFT", "AKAZE"]
                        and self.template_img is not None
                    ):
                        self.template_kp, self.template_des = (
                            self.detector.detectAndCompute(self.template_img, None)
                        )

            except Exception as e:
                self.log_signal.emit(f"Worker update error: {e}")

    @Slot(dict)
    def update_contour_params(self, params):
        with QMutexLocker(self.mutex):
            self.contour_params.update(params)

    @Slot(str)
    def set_ssim_reference(self, file_path):
        with QMutexLocker(self.mutex):
            if not file_path:
                self.ssim_ref_img = None
                self.ssim_cache = None
                if self.current_algo == 'SSIM':
                    self.is_matching_enabled = False
                return
            
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.ssim_ref_img = img
                self.ssim_cache = None
                self.log_signal.emit(f"Worker: SSIM reference loaded {file_path}")
                if self.current_algo == 'SSIM':
                    self.is_matching_enabled = True
            else:
                self.log_signal.emit("Worker: Failed to load SSIM reference")

    @Slot(str)
    def set_template(self, file_path):
        with QMutexLocker(self.mutex):
            if not file_path:
                if self.current_algo != "ARUCO" and self.current_algo != "QRCODE" and self.current_algo != "SSIM":
                    self.is_matching_enabled = False
                self.template_img = None
                return

            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.template_img = img
                if self.detector is not None and self.current_algo not in [
                    "ARUCO",
                    "QRCODE",
                    "SSIM",
                ]:
                    self.template_kp, self.template_des = (
                        self.detector.detectAndCompute(img, None)
                    )
                    if self.template_des is not None:
                        self.is_matching_enabled = True
                        self.log_signal.emit(f"Worker: Template loaded {file_path}")
                    else:
                        self.log_signal.emit("Worker: No features in template")
                else:
                    self.log_signal.emit(
                        f"Worker: Template loaded (not used for {self.current_algo})"
                    )
            else:
                self.log_signal.emit("Worker: Failed to load template")

    @Slot(bool)
    def toggle_matching(self, enabled):
        with QMutexLocker(self.mutex):
            self.is_matching_enabled = enabled

    @Slot(bool)
    def toggle_contours(self, enabled):
        with QMutexLocker(self.mutex):
            self.is_contours_enabled = enabled

    @Slot(int, int, int, int, bytes)
    def process_frame(self, width, height, bytes_per_line, fmt, data_bytes):
        # This runs in the worker thread
        with QMutexLocker(self.mutex):
            matching_active = self.is_matching_enabled
            contours_active = self.is_contours_enabled
            algo = self.current_algo

            # Local refs
            local_detector = self.detector
            local_bf = self.bf
            local_template_des = self.template_des
            local_template_kp = self.template_kp
            local_template_img = self.template_img
            local_aruco_dict = self.aruco_dict
            local_aruco_params = self.aruco_params
            local_aruco_obj = self.aruco_obj
            local_aruco_display = self.aruco_display.copy()
            local_contour_params = self.contour_params.copy()
            local_qr_detector = self.qr_detector
            local_ssim_ref = self.ssim_ref_img
            local_ssim_cache = self.ssim_cache
            local_params = self.last_params.copy()

        try:
            channels = bytes_per_line // width
            img_np = np.frombuffer(data_bytes, dtype=np.uint8).reshape(
                (height, width, channels)
            )

            # If no processing is needed, return original
            if not matching_active and not contours_active:
                qimg = QImage(
                    img_np.data, width, height, bytes_per_line, QImage.Format(fmt)
                ).copy()
                self.result_ready.emit(qimg)
                return

            # Prepare visualization image (BGR for OpenCV drawing)
            if channels == 1:
                vis_img = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                gray_frame = img_np
            else:
                vis_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                gray_frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            # --- 1. Contours Processing ---
            if contours_active:
                # Blur to reduce noise
                blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)

                # Binarize based on mode
                if local_contour_params.get("mode") == "Threshold":
                    # Binary Threshold
                    _, binary_img = cv2.threshold(
                        blurred,
                        local_contour_params.get("threshold", 127),
                        255,
                        cv2.THRESH_BINARY,
                    )
                    edges = binary_img  # Treat binary result as input for findContours
                else:
                    # Default: Canny Edges
                    edges = cv2.Canny(
                        blurred,
                        local_contour_params["thresh_min"],
                        local_contour_params["thresh_max"],
                    )

                # Find Contours
                contours, _ = cv2.findContours(
                    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # Draw on vis_img
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if (
                        local_contour_params["min_area"]
                        < area
                        < local_contour_params["max_area"]
                    ):
                        thickness = -1 if local_contour_params["fill"] else 2
                        cv2.drawContours(vis_img, [cnt], -1, (0, 255, 0), thickness)
                        if local_contour_params["box"]:
                            x, y, w, h = cv2.boundingRect(cnt)
                            cv2.rectangle(
                                vis_img, (x, y), (x + w, y + h), (0, 0, 255), 2
                            )

            # --- 2. Matching Processing ---
            if matching_active:
                if algo == 'SSIM' and local_ssim_ref is not None:
                    # Resize reference if dimensions don't match or cache is missing
                    use_cache = local_ssim_cache
                    
                    if use_cache is None or use_cache["img"].shape != gray_frame.shape:
                        if local_ssim_ref.shape != gray_frame.shape:
                             resized_ref = cv2.resize(local_ssim_ref, (gray_frame.shape[1], gray_frame.shape[0]))
                        else:
                             resized_ref = local_ssim_ref
                        
                        use_cache = precompute_ssim_constants(resized_ref)
                        
                        # Update cache if ref hasn't changed
                        with QMutexLocker(self.mutex):
                            if self.ssim_ref_img is local_ssim_ref:
                                self.ssim_cache = use_cache
                    
                    score = compute_ssim_cached(gray_frame, use_cache)
                    self.ssim_score_signal.emit(score)
                    
                    cv2.putText(vis_img, f"SSIM: {score:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                elif algo == "QRCODE" and local_qr_detector:
                    retval, decoded_info, points, straight_qrcode = (
                        local_qr_detector.detectAndDecodeMulti(gray_frame)
                    )
                    if retval:
                        # points is a list of points for each QR code
                        for i in range(len(decoded_info)):
                            text = decoded_info[i]
                            pts = points[i].astype(int)

                            # Draw bounding box
                            for j in range(4):
                                cv2.line(
                                    vis_img,
                                    tuple(pts[j]),
                                    tuple(pts[(j + 1) % 4]),
                                    (255, 0, 0),
                                    2,
                                )

                            # Draw text
                            cv2.putText(
                                vis_img,
                                text,
                                tuple(pts[0]),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 0, 0),
                                2,
                            )

                            if text:
                                self.qr_found_signal.emit(text)

                elif algo == "ARUCO" and local_aruco_obj:
                    corners, ids, rejected = local_aruco_obj.detectMarkers(gray_frame)
                    if local_aruco_display["rejected"] and rejected:
                        cv2.aruco.drawDetectedMarkers(
                            vis_img, rejected, borderColor=(100, 0, 255)
                        )
                    if ids is not None and len(ids) > 0:
                        display_ids = ids if local_aruco_display["ids"] else None
                        cv2.aruco.drawDetectedMarkers(vis_img, corners, display_ids)

                elif algo == "HOUGH_CIRCLE":
                    # Hough Circles
                    # Use gray_frame
                    # Apply blur to reduce noise
                    blur_gray = cv2.medianBlur(gray_frame, 5)
                    
                    circles = cv2.HoughCircles(
                        blur_gray,
                        cv2.HOUGH_GRADIENT,
                        dp=local_params.get("dp", 1.0),
                        minDist=local_params.get("minDist", 50.0),
                        param1=local_params.get("param1", 100.0),
                        param2=local_params.get("param2", 30.0),
                        minRadius=int(local_params.get("minRadius", 1)),
                        maxRadius=int(local_params.get("maxRadius", 100))
                    )
                    
                    if circles is not None:
                        circles = np.uint16(np.around(circles))
                        for i in circles[0, :]:
                            # draw the outer circle
                            cv2.circle(vis_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                            # draw the center of the circle
                            cv2.circle(vis_img, (i[0], i[1]), 2, (0, 0, 255), 3)

                elif (
                    algo in ["ORB", "SIFT", "AKAZE"]
                    and local_template_img is not None
                    and local_detector is not None
                ):
                    # Always attempt detection
                    kp_frame, des_frame = local_detector.detectAndCompute(
                        gray_frame, None
                    )
                    
                    # Ensure iterable if None
                    if kp_frame is None: kp_frame = []

                    good_matches = []
                    if local_template_des is not None and des_frame is not None:
                        matches = local_bf.match(local_template_des, des_frame)
                        matches = sorted(matches, key=lambda x: x.distance)
                        good_matches = matches[:20]

                    # Always draw, even if no matches, to show template and keypoints
                    # Using flags=0 allows drawing single points (unmatched keypoints)
                    vis_img = cv2.drawMatches(
                        local_template_img,
                        local_template_kp if local_template_kp else [],
                        vis_img,
                        kp_frame,
                        good_matches,
                        None,
                        flags=0 
                    )

            # Convert final result to QImage
            # vis_img is BGR (or BGR-like output from drawMatches)
            # Need RGB for QImage
            res_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            h, w, c = res_img_rgb.shape
            qimg = QImage(res_img_rgb.data, w, h, w * c, QImage.Format_RGB888).copy()
            self.result_ready.emit(qimg)

        except Exception as e:
            self.log_signal.emit(f"Worker processing error: {e}")
