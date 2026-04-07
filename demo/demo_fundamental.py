# run in lightglue conda environment!!!

from PIL import Image
import torch
import cv2
from romatch import roma_outdoor
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="assets/sacre_coeur_A.jpg", type=str)
    parser.add_argument("--im_B_path", default="assets/sacre_coeur_B.jpg", type=str)

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path

    # Create model
    roma_model = roma_outdoor(device=device)


    W_A, H_A = Image.open(im1_path).size
    W_B, H_B = Image.open(im2_path).size

    # Match
    warp, certainty = roma_model.match(im1_path, im2_path, device=device)
    # print('certainty', certainty)
    certainty_cpu = certainty.cpu().numpy().squeeze()
    print('certainty shape, dtype', certainty_cpu.shape, certainty_cpu.dtype)
    print('certainty stats: min {:.4f}, max {:.4f}, mean {:.4f}'.format(certainty_cpu.min(), certainty_cpu.max(), certainty_cpu.mean()))
    # convert certainty to image for visualization
    certainty1 = certainty_cpu[:, :certainty_cpu.shape[1]//2]
    certainty1_resized = cv2.resize(certainty1, (W_A, H_A), interpolation=cv2.INTER_NEAREST)
    certainty2 = certainty_cpu[:, certainty_cpu.shape[1]//2:]
    certainty2_resized = cv2.resize(certainty2, (W_B, H_B), interpolation=cv2.INTER_NEAREST)
    certainty_img = (certainty_cpu * 255).astype(np.uint8)
    certainty1_img = (certainty1_resized * 255).astype(np.uint8)
    certainty2_img = (certainty2_resized * 255).astype(np.uint8)
    # concat the certainty maps side by side
    max_y = max(certainty1_img.shape[0], certainty2_img.shape[0])
    certainty1_padded = cv2.copyMakeBorder(certainty1_img, 0, max_y - certainty1_img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
    certainty2_padded = cv2.copyMakeBorder(certainty2_img, 0, max_y - certainty2_img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
    certainty_concat = np.hstack((certainty1_padded, certainty2_padded))
    certainty_concat = cv2.cvtColor(certainty_concat, cv2.COLOR_GRAY2BGR)
    cv2.imshow('certainty', certainty_concat)
    cv2.waitKey(1000000)
    cv2.destroyAllWindows()
    cv2.imwrite('certainty_map.png', certainty_concat)
    # Sample matches for estimation
    matches, certainty = roma_model.sample(warp, certainty)

    kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)    
    F, mask = cv2.findFundamentalMat(
        kpts1.cpu().numpy(), kpts2.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
    )
    # certainty = certainty.cpu().numpy()
    # print('certainty', certainty)
    # visualize inlier matches
    im1 = cv2.imread(im1_path)
    im2 = cv2.imread(im2_path)
    inlier_matches = matches[mask.ravel() == 1]
    inlier_kpts1 = kpts1
    inlier_kpts2 = kpts2
    # print('inlier kpts1', inlier_kpts1)
    # print('inlier kpts2', inlier_kpts2)
    # convert to numpy and draw matches
    inlier_kpts1 = inlier_kpts1.cpu().numpy()
    inlier_kpts2 = inlier_kpts2.cpu().numpy()
    # print('inlier matches', inlier_matches)
    print('inlier_kpts1 shape, dtype', inlier_kpts1.shape, inlier_kpts1.dtype)
    # num_viz = min(5000, len(inlier_kpts1))
    num_viz = len(inlier_kpts1)
    sample_idx = np.random.choice(len(inlier_kpts1), num_viz, replace=False)
    pts1 = inlier_kpts1[sample_idx]  # (N, 2)
    pts2 = inlier_kpts2[sample_idx]  # (N, 2)
    # Build side-by-side canvas with only keypoints drawn
    canvas_w = im1.shape[1] + im2.shape[1]
    canvas_h = max(im1.shape[0], im2.shape[0])

    def build_base_canvas():
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:im1.shape[0], :im1.shape[1]] = im1
        canvas[:im2.shape[0], im1.shape[1]:] = im2
        # use random colors for each match
        
        for x, y in pts1:
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            cv2.circle(canvas, (int(x), int(y)), 1, color, -1)
        # for x, y in pts2:
        #     cv2.circle(canvas, (int(x) + im1.shape[1], int(y)), 1, (0, 255, 0), -1)
        return canvas

    base_canvas = build_base_canvas()

    weighted_canvas = base_canvas.astype(np.float32) * np.sqrt(certainty_concat.astype(np.float32) / 255.0)
    weighted_canvas = weighted_canvas.astype(np.uint8)
    cv2.imshow('matches', weighted_canvas)
    cv2.waitKey(1000000)

    display = base_canvas.copy()
    conf_display = certainty_concat.copy()
    CLICK_RADIUS = 10

    def on_mouse(event, x, y, flags, param):
        global display
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # Determine which image was clicked
        if x < im1.shape[1]:
            # Click on left image — find nearest keypoint in pts1
            dists = np.hypot(pts1[:, 0] - x, pts1[:, 1] - y)
            idx = int(np.argmin(dists))
            if dists[idx] > CLICK_RADIUS:
                return
            display = base_canvas.copy()
            conf_display = certainty_concat.copy()
            p1 = (int(pts1[idx, 0]), int(pts1[idx, 1]))
            p2 = (int(pts2[idx, 0]) + im1.shape[1], int(pts2[idx, 1]))
            conf1 = certainty1_resized[int(pts1[idx, 1]), int(pts1[idx, 0])]
            conf2 = certainty2_resized[int(pts2[idx, 1]), int(pts2[idx, 0])]
            print(f"Conf1: {conf1:.4f}, Conf2: {conf2:.4f}")
        else:
            # Click on right image — find nearest keypoint in pts2
            rx = x - im1.shape[1]
            dists = np.hypot(pts2[:, 0] - rx, pts2[:, 1] - y)
            idx = int(np.argmin(dists))
            if dists[idx] > CLICK_RADIUS:
                return
            display = base_canvas.copy()
            conf_display = certainty_concat.copy()
            p1 = (int(pts1[idx, 0]), int(pts1[idx, 1]))
            p2 = (int(pts2[idx, 0]) + im1.shape[1], int(pts2[idx, 1]))
            conf2 = certainty2_resized[int(pts2[idx, 1]), int(pts2[idx, 0])]
            conf1 = certainty1_resized[int(pts1[idx, 1]), int(pts1[idx, 0])]
            print(f"Conf1: {conf1:.4f}, Conf2: {conf2:.4f}")

        cv2.circle(display, p1, 5, (0, 0, 255), -1)
        cv2.circle(display, p2, 5, (0, 0, 255), -1)
        cv2.line(display, p1, p2, (0, 0, 255), 1)

        cv2.circle(conf_display, p1, 5, (0, 0, 255), 1)
        cv2.putText(conf_display, f"{conf1:.4f}", (p1[0]+5, p1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.circle(conf_display, p2, 5, (0, 0, 255), 1)
        cv2.putText(conf_display, f"{conf2:.4f}", (p2[0]+5, p2[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow('matches', display)
        cv2.imshow('certainty', conf_display)

    cv2.namedWindow('matches')
    cv2.setMouseCallback('matches', on_mouse)
    cv2.imshow('matches', display)
    cv2.imshow('certainty', conf_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()