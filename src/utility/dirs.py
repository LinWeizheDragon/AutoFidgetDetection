import os

def create_dirs(createdirs):
    try:
        for dir in createdirs:
            if not os.path.exists(dir):
                os.makedirs(dir)
                print("{} folder created".format(dir))
            else:
                print("folder already exists: {}".format(dir))
    except Exception as e:
        print(e)