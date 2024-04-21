import SKB
import ASI
import SC_with_ASC
if __name__ == '__main__':
    image_path = "images/1.png"
    SKB.SKB_with_auto(image_path)
    ASI.semantic_aware_images_generation(image_path.replace(".png",""))
    SC_with_ASC.data_transmission(".")

