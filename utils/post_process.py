import json
def postprocess(path):
    '''
    filter category 1, keep category 0; save to json file
    return: json file path
    '''
    with open(path, "r") as f:
        data = json.load(f)
        print("Number of images before postprocessing:", len(data))
        # for each element in imgs, if category is 1, remove it
        for img in data:
            if img['category_id'] == 1:
                data.remove(img)
        print("Number of images after postprocessing:", len(data))
        # save to json file
        path = path.replace('.json', '_postprocess.json')
        with open(path, "w") as f:
            json.dump(data, f)
        print("Saved to", path)
    return path        
def main():
    json_path = '/data/baby/Workspace/huyenbk/iai-baby/yolov7/runs/test/tiny-_2labels_loosen0.2byhead/best_predictions.json'
    postprocess(json_path)
    pass

if __name__ == "__main__":
    main()