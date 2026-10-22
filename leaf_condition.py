from PIL import Image
import numpy as np


def check_leaf_condition(image_path):
    """
    Checks whether the uploaded leaf is:
    - DRY / WILTED
    - NOT_DRY / FRESH

    Returns:
        status
        confidence
        message
    """

    try:
        # Open image
        image = Image.open(image_path).convert("RGB")

        # Resize for faster processing
        image = image.resize((256, 256))

        img = np.array(image).astype(np.float32) / 255.0

        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        # --------------------------------------------------
        # GREEN PIXEL DETECTION
        # --------------------------------------------------
        green_mask = (
            (g > r * 1.05) &
            (g > b * 1.05) &
            (g > 0.20)
        )

        green_ratio = np.mean(green_mask)

        # --------------------------------------------------
        # BROWN / DRY PIXEL DETECTION
        # --------------------------------------------------
        brown_mask = (
            (r > g * 1.15) &
            (g > b * 1.05) &
            (r > 0.20) &
            (g > 0.12)
        )

        brown_ratio = np.mean(brown_mask)

        # --------------------------------------------------
        # YELLOW / DRY PIXEL DETECTION
        # --------------------------------------------------
        yellow_mask = (
            (r > 0.45) &
            (g > 0.35) &
            (b < g * 0.80)
        )

        yellow_ratio = np.mean(yellow_mask)

        # --------------------------------------------------
        # BRIGHTNESS
        # --------------------------------------------------
        brightness = np.mean(img)

        # --------------------------------------------------
        # DRYNESS SCORE
        # --------------------------------------------------
        dryness_score = (
            (brown_ratio * 0.50) +
            (yellow_ratio * 0.20) +
            ((1 - green_ratio) * 0.30)
        )

        # Convert to percentage
        dryness_percentage = min(
            100,
            max(0, dryness_score * 100)
        )

        # --------------------------------------------------
        # DECISION
        # --------------------------------------------------

        if green_ratio < 0.12 and dryness_score > 0.50:

            return {
                "status": "DRY",
                "confidence": round(dryness_percentage, 2),
                "message": "இந்த இலை காய்ந்த நிலையில் உள்ளது."
            }

        elif green_ratio < 0.20 and dryness_score > 0.42:

            return {
                "status": "DRY",
                "confidence": round(dryness_percentage, 2),
                "message": "இந்த இலை காய்ந்த அல்லது வாடிய நிலையில் இருக்கலாம்."
            }

        else:

            freshness = 100 - dryness_percentage

            return {
                "status": "NOT_DRY",
                "confidence": round(freshness, 2),
                "message": "இந்த இலை காய்ந்த நிலையில் இல்லை. Image processing தொடரலாம்."
            }

    except Exception as e:

        return {
            "status": "ERROR",
            "confidence": 0,
            "message": f"Image processing error: {str(e)}"
        }