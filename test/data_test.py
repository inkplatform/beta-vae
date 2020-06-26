import os

path = '../data/casia' # casia, feret,
fileList = os.listdir(path)

if os.path.exists("casia_dest.txt"):
    os.remove("casia_dest.txt")

file_write_obj = open("casia_dest.txt", 'w')

for file in sorted(fileList):
    #print(file)
    img_list = os.listdir(path + '/' + file)
    img_num = len(img_list)
    i = 0

    for img in sorted(img_list):
        print(img)
        if not img.endswith('.db'):
            i= i + 1
            if i < (int(img_num * 0.8)):
                file_write_obj.writelines(file+'/'+img + ',0')
            elif i >= (int(img_num * 0.8)) and i < (int(img_num * 0.9)):
                file_write_obj.writelines(file+'/'+img + ',1')
            else:
                file_write_obj.writelines(file+'/'+img + ',2')
            file_write_obj.write('\n')


'''with open('../data/list_eval_partition.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    final_list = []
    next(reader)
    i = 0
    for row in reader:
        im_id, category = row
        i = i+1
        if im_id not in fileList:
            print(im_id)
'''

# print(set(fileList).symmetric_difference(set(final_list)))