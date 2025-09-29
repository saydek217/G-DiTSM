#!/usr/bin/env python3
# ---------------------------------------------------------------
# Real-time Gesture Inference with mobilenetv3_deptheca_mega
# ---------------------------------------------------------------
import time, sys, argparse
import torch, cv2, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# ─────────── CLI ───────────────────────────────────────────────
def cli():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='best.pth.tar')
    p.add_argument('--labels', default='datas/jester/category.txt')
    p.add_argument('--cam', type=int, default=0)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--no-fp16', action='store_true')
    p.add_argument('--ema-alpha', type=float, default=0.2)
    p.add_argument('--dwell-in', type=int, default=4)
    p.add_argument('--dwell-out', type=int, default=8)
    p.add_argument('--thr-in', type=float, default=0.70)
    p.add_argument('--thr-out', type=float, default=0.55)
    p.add_argument('--num-segments', type=int, default=8)
    return p.parse_args()

# ─────────── Load Model ────────────────────────────────────────
def load_model(ckpt_path, arch, num_classes, num_segments, device, use_fp16):
    from ops.models import TSN  # Import the training-time wrapper
    from archs.mobilenet_v3_deptheca_mega import MobileNetV3_DepthECAmega  # This must be the one used during training

    # Instantiate exactly like training
    model = TSN(
        num_classes,
        num_segments=num_segments,
        base_model='mobilenetv3_deptheca_mega',
        dropout=0.0,
        partial_bn=True
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)

    # Clean state_dict
    clean_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        clean_dict[new_key] = v

    missing, unexpected = model.load_state_dict(clean_dict, strict=False)
    print(f"✅ Model loaded with {len(missing)} missing keys, {len(unexpected)} unexpected keys.")
    if missing: print("  → Missing:", missing[:5], "..." if len(missing) > 5 else "")
    if unexpected: print("  → Unexpected:", unexpected[:5], "..." if len(unexpected) > 5 else "")

    model.to(device).eval()
    if use_fp16:
        model.half()

    return model


# ─────────── Stable Gesture Filter ─────────────────────────────
class StableGesture:
    def __init__(self, labels, d_in, d_out, t_in, t_out):
        self.labels = labels
        self.d_in, self.d_out = d_in, d_out
        self.t_in, self.t_out = t_in, t_out
        self.cur = "No gesture"
        self.cnt = 0

    def update(self, lab, prob):
        if self.cur == "No gesture":
            if lab != "No gesture" and prob >= self.t_in:
                self.cnt += 1
                if self.cnt >= self.d_in: self.cur, self.cnt = lab, 0
            else: self.cnt = 0
        else:
            if lab == self.cur and prob >= self.t_out:
                self.cnt = 0
            else:
                self.cnt += 1
                if self.cnt >= self.d_out: self.cur, self.cnt = "No gesture", 0
        return self.cur

# ─────────── Main ──────────────────────────────────────────────
def main():
    args = cli()
    labels = [l.strip() for l in open(args.labels, encoding='utf-8') if l.strip()]
    dev = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                       else args.device if args.device != 'auto' else 'cpu')
    use_fp16 = dev.type == 'cuda' and not args.no_fp16

    print(f"\n✅ Loading model '{args.ckpt}' on {dev}")
    net = load_model(args.ckpt, 'mobilenetv3_deptheca_mega', len(labels), args.num_segments, dev, use_fp16)

    tfm = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(), T.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened(): sys.exit("❌ Camera error")
    print("🎥 Press 'q' to quit")

    tracker = StableGesture(labels, args.dwell_in, args.dwell_out,
                            args.thr_in, args.thr_out)
    ema, prev = None, time.time()
    frame_queue = []

    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok: continue
            frame = cv2.resize(frame, (1280, 720))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = tfm(Image.fromarray(rgb)).unsqueeze(0).to(dev)
            if use_fp16: tensor = tensor.half()

            frame_queue.append(tensor)
            if len(frame_queue) < args.num_segments:
                continue
            if len(frame_queue) > args.num_segments:
                frame_queue.pop(0)

            input_clip = torch.cat(frame_queue, dim=0).unsqueeze(0)  # (1, T, C, H, W)
            input_clip = input_clip.view(1, args.num_segments, 3, 224, 224)

            output = net(input_clip.to(dev))
            if use_fp16: output = output.float()

            probs = F.softmax(output, dim=1)[0]
            ema = probs if ema is None else (1 - args.ema_alpha) * ema + args.ema_alpha * probs
            conf, idx = ema.max(0); prob = conf.item(); label = labels[idx]

            disp = tracker.update(label, prob)
            now = time.time(); fps = 1 / (now - prev); prev = now

            cv2.putText(frame, f"{disp}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (0, 255, 0) if disp != "No gesture" else (100, 100, 100), 3)
            cv2.putText(frame, f"{prob:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            cv2.putText(frame, f"{fps:4.1f} FPS", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            cv2.imshow("🖐️ Gesture Detection", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

    cap.release(); cv2.destroyAllWindows()

# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
