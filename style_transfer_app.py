import torch
import cv2
import numpy as np
from PIL import Image
from fast_neural_style.transformer_net import TransformerNet

# ----------------------------
# 1. Load Style Models
# ----------------------------
styles = ['candy', 'mosaic', 'udnie']
style_paths = {s: f"models/{s}.pth" for s in styles}
style_models = {}

print("Loading style models...")
for s in styles:
    model = TransformerNet()
    state_dict = torch.load(style_paths[s])
    # Remove deprecated running stats if present
    for k in list(state_dict.keys()):
        if k.endswith('running_mean') or k.endswith('running_var'):
            del state_dict[k]
    model.load_state_dict(state_dict)
    model.eval()
    style_models[s] = model
print("Models loaded!")

current_style = 'candy'

# ----------------------------
# 2. Preprocessing & Postprocessing
# ----------------------------
def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame)
    img_tensor = torch.tensor(np.array(pil_img)).permute(2, 0, 1).float()
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def postprocess(tensor):
    img = tensor.squeeze().permute(1, 2, 0).detach().numpy()
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

# ----------------------------
# 3. Webcam Loop
# ----------------------------
cap = cv2.VideoCapture(0)
snapshot_count = 0
print("Press 'q' to quit, 's' to save snapshot, '1-3' to switch styles.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_tensor = preprocess(frame)
    with torch.no_grad():
        output_tensor = style_models[current_style](img_tensor)

    output_image = postprocess(output_tensor)

    cv2.putText(output_image, f"Style: {current_style}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Style Transfer Webcam", output_image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        snapshot_count += 1
        cv2.imwrite(f"snapshot_{snapshot_count}.jpg", output_image)
        print(f"Saved snapshot_{snapshot_count}.jpg")
    elif key == ord('1'):
        current_style = styles[0]
        print(f"Switched to style: {current_style}")
    elif key == ord('2'):
        current_style = styles[1]
        print(f"Switched to style: {current_style}")
    elif key == ord('3'):
        current_style = styles[2]
        print(f"Switched to style: {current_style}")

cap.release()
cv2.destroyAllWindows()
