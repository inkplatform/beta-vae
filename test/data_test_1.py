import os

path = '../data/feret' # casia, feret,
fileList = os.listdir(path)

if os.path.exists("feret_dest.txt"):
    os.remove("feret_dest.txt")

file_write_obj = open("feret_dest.txt", 'w')

i = 0
for file in sorted(fileList):
    #print(file)
    img_list = os.listdir(path + '/' + file)
    img_num = len(img_list)
    if os.path.isfile(img_list):
        print('true')
        print(int(img_num * 0.8))
        print(int(img_num * 0.9))
    #print(img_num)
    '''for img in sorted(img_list):
        i= i + 1
        if i < 162771:
            file_write_obj.writelines(file+'/'+img + ',0')
        elif i > 162771 and i < 182638:
            file_write_obj.writelines(file+'/'+img + ',1')
        else:
            file_write_obj.writelines(file+'/'+img + ',2')
        file_write_obj.write('\n')'''