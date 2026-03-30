import sys
from .bindings import MindVisionCamera, VideoThread


def main():
    print("Successfully imported _mindvision_qobject_py module.")

    print("\nTesting MindVisionCamera:")
    camera = MindVisionCamera()
    print(f"MindVisionCamera instance created: {camera}")

    camera.open()
    print(f"Initial AutoExposure: {camera.getAutoExposure()}")
    camera.setAutoExposure(True)
    print(f"AutoExposure after setting to True: {camera.getAutoExposure()}")
    camera.setAutoExposure(False)

    min_exp, max_exp = camera.getExposureTimeRange()
    print(f"Exposure Time Range: min={min_exp}ms, max={max_exp}ms")

    min_exp, max_exp = camera.getExposureTimeRange()
    step_exp = camera.getExposureTimeStep()

    print(f"Exposure Time Range: {min_exp} ms to {max_exp} ms")
    print(f"Exposure Time Step: {step_exp} ms")

    print("\nTesting AE Target:")
    try:
        ae_target = camera.getAeTarget()
        print(f"Initial AE Target: {ae_target}")

        new_target = 100 if ae_target != 100 else 120
        print(f"Setting AE Target to: {new_target}")
        if camera.setAeTarget(new_target):
            print(f"Set AE Target success. New value: {camera.getAeTarget()}")
        else:
            print("Set AE Target failed.")
    except AttributeError:
        print("AE Target methods not found on object.")

    if step_exp > 0:
        print("Calculating and verifying ALL valid exposure times (this may take a moment)...")
        current = min_exp
        match_count = 0
        camera.setAutoExposure(False)

        while current <= max_exp:
            camera.setExposureTime(current)

            actual = camera.getExposureTime()
            if abs(actual - current) > 0.001:
                print(f"    MISMATCH: Set {current:.4f}, Got {actual:.4f}")
            else:
                match_count += 1

            current += step_exp

        print(f"Verification complete. Total valid (matched) exposure values: {match_count}")
    else:
        print("Step is 0 or invalid, cannot calculate steps.")

    print("\nTesting VideoThread:")
    video_thread = VideoThread()
    print(f"VideoThread instance created: {video_thread}")

    print("Attempting to call startRecording (dummy parameters)...")
    video_thread.startRecording(640, 480, 30.0, "test_video.avi")
    print("startRecording called.")

    print("Attempting to call stopRecording...")
    video_thread.stopRecording()
    print("stopRecording called.")

    print(f"Is running: {video_thread.isRunning()}")
    print("\nPython wrapper test completed successfully (syntactically).")


if __name__ == "__main__":
    sys.exit(main())
