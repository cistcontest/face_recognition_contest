import os 
from api_detect import main as d_main
from api_out import main as o_main


CONTAINER_DIR = "/home/thanhnn/dataset/QH_API_160"
RAW_DIR = "/home/thanhnn/dataset/QH/"

if __name__ == "__main__":
    output_dir = os.path.join(CONTAINER_DIR, "Extracted")
    input_dir = RAW_DIR
    gpu_memory_fraction = 1.0
    detect_multiple_faces = True
    margin = 32
    image_size = 160
    random_order = False
    test_image_path = "/home/thanhnn/dataset/QH/000_quang_hai.jpg"
    
    print("Start detect")
    bounding_boxes_filename, extracted_dir, test_output_filename = d_main(output_dir=output_dir,
                                    input_dir=input_dir,
                                    gpu_memory_fraction=gpu_memory_fraction,
                                    random_order=random_order,
                                    detect_multiple_faces=detect_multiple_faces,
                                    margin=margin,
                                    image_size=image_size,
                                    test_image_path=test_image_path)

    print("Start recog")
    model_path = "/home/thanhnn/facenet/checkpoints/20180402-114759"    
    data_dir = extracted_dir
    base_data_dir = os.path.join(RAW_DIR, "quang_hai")
    output_data_dir = os.path.join(CONTAINER_DIR, "Output")

    if not os.path.exists(test_output_filename):
        raise Exception("test output filename not found")

    test_path = test_output_filename
    bb_path = bounding_boxes_filename
    result_path = os.path.join(CONTAINER_DIR, "result.txt")
    final_path = os.path.join(CONTAINER_DIR, "final.txt")

    o_main(model_path=model_path,
            data_dir=data_dir,
            base_data_dir=base_data_dir,
            output_data_dir=output_data_dir,
            test_path=test_path,
            bb_path=bb_path,
            result_path=result_path,
            final_path=final_path)