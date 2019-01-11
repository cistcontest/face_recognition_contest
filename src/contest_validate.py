import os


GT_FILE = "/home/thanhnn/dataset/QH_API_160/gt.txt"
RESULT_FILE = "/home/thanhnn/dataset/QH_API_160/final.txt"


def evaluate():
    result_content = None
    gt_content = None
    with open(RESULT_FILE, "r") as f:
        lines = f.readlines()
        result_content = [x.strip() for x in lines]
        result_content = sorted(result_content)

    with open(GT_FILE, "r") as f:
        lines = f.readlines()
        gt_content = [x.strip() for x in lines]
        gt_content = sorted(gt_content)
    
    incorrect = 0
    too_many = 0
    for i in gt_content:
        gt_element = i.split(",")
        meet = [x for x in result_content if x.split(",")[0] == gt_element[0]]
        # print("len meet: %d" % len(meet))
        # print(meet)
        if len(meet) > 1:
            incorrect += 1
            too_many += 1
        elif len(meet) < 1:
            incorrect += 1
        else:
            pred = meet[0].split(",")
            t1, l1, b1, r1 = gt_element[1], gt_element[2], gt_element[3], gt_element[4]
            t2, l2, b2, r2 = pred[2], pred[1], pred[4], pred[3]
            t1, l1, b1, r1 = int(t1), int(l2), int(b1), int(r1)
            t2, l2, b2, r2 = int(t2), int(l2), int(b2), int(r2)
            t3 = max(t1,t2)
            l3 = max(l1, l2)
            b3 = min(b1, b2)
            r3 = min(r1, r2)
            s_insec = (b3-t3) * (r3-l3)
            s1 = (b1-t1) * (r1-l1)
            s2 = (b2-t2) * (r2-l2)
            s_union = float(s1 + s2 - s_insec)
            iou = float(s_insec) / s_union
            print("IOU: %.2f" % iou)
            if 1 >= iou > 0.5:
                # print("  + overlap > 0.5")
                pass
            else:
                incorrect += 1
    
    print("Total excess: %d" % too_many)
    print("Total incorrect: %d" % incorrect)
    print("Total correct: %d" % (len(gt_content) - incorrect))
    print("Total: %d" % len(gt_content))



if __name__ == "__main__":
    evaluate()