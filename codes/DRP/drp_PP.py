import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import os, re, glob
import cv2
import csv
import math
import csv
import datetime


def damagedRAM_serverDataPrep(csv_path, server, write_path, protocol, skipped, debug_level, p2, min_dist, r_inc, p_inc, c_inc, from_center):
    os.mkdir(write_path + "good")
    os.mkdir(write_path + "newly_skipped")
    os.mkdir(write_path + "pp_off")

    print("Current protocol:", protocol)
    if protocol != "NA":
      metlog = pd.read_excel(csv_path, skiprows=[0, 1])
      #print(metlog.head())
      #print("Full metadata shape:", metlog.shape)
      metlog = metlog[metlog['Measurement Type'] == 'Sites']
      metlog = metlog[metlog['Lighting'] == 'Coax']
      metlog = metlog[metlog['Date'] > datetime.datetime(2023, 1, 1)]
      metlog = metlog[metlog['Comment'] == protocol]#.str.contains(protocol)]
      print("Filtered metadata head:", metlog.head())
      print("Filtered metadata tail:", metlog.tail())
      print("Filtered metadata shape:", metlog.shape)
      #print(metlog.iloc[0, :]['Comment'].split('RAM')) add in RAM360
      protocol_num = []
      protocol_num = metlog.iloc[0, :]['Comment'].split('RAM')[-1]
      print("pc num", protocol_num)

      #if len(protocol_num == 3):

      protocol_count = metlog.shape[0]
      print("Count of current protocol directories:", protocol_count)

      all_ims = []

      for i in range(protocol_count):


        metid = metlog["Metrology ID"].iloc[i]
        metidstr = str(int(metid))
        year = 2000 + int(metidstr[:2])
        month = int(metidstr[2:4])
        day = int(metidstr[4:6])

        date = datetime.datetime(year, month, day)
        path = date.strftime(server + 'Microscopy/%Y/%B %Y/%B %d %Y/'+ metidstr + '/*/')
        all_ims += glob.glob(path + "*.tif")
      print("Count of total images found:", len(all_ims))
    else:
      with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        attributes = next(reader)
        part_meta = list(reader)  # np.array(list(reader))
      all_ims = part_meta[0] 
    new_ims = []
    RAM_labels, RAMEO_labels = [], []

    #print("Ready to read ", len(part_meta), " images from metadata links in CSV provided.")
    #protocol, sub_dir = None, None
    j = 0

    # leftovers
    add_another = False
    new_skipped = []
    # if not found
    crop = 'nc'
    tried_crop = False
    second_crop = False
    tried_p2 = False
    tried_p2b = False
    # max pixel protocol
    maxpp = 530
    class_dict = {"No Damage": 0, "Damage": 1}
    
    while j < len(all_ims): # may not inc if havent extracted RAM c, r
        dclass =0
        if protocol == "NA":
          dclass = class_dict[part_meta[j][-1]]
          protocol_num = part_meta[j][1].split('RAM')[-1]  
        print("Checking image #: ", j)
        print("name: ", all_ims[j])
        pixel_protocols = []

        if protocol != "NA" and j == 5000:
            break

        pixel_protocols = []
        # resolution is always part of image name before .tif
        res = float(all_ims[j].split("/")[-1][-8:-4])
        #print("res", res)
         
        for i in range(1):#len(protocol_num) - 1):
            pixel_protocols.append(float(protocol_num) / res)
            #print(pixel_protocols)


        im_path = all_ims[j]
        im = cv2.imread(im_path)


        #res = 1
        # print("class", class_dict[part_meta[j][-4]])
        if j in skipped:
            RAMEO_count = 0
            x, y = im.shape[1] // 2, im.shape[0] // 2
            if np.min(im) < 60:
                # print(j)
                mask = np.zeros_like(im)
                r = int(pixel_protocols[0] // 2 + r_inc + 20)
                if j < 300:
                    r += 10
                mask = cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
                masked_im = cv2.bitwise_and(im, mask)
                both = np.concatenate((im, masked_im), axis=1)
                cv2.imwrite(write_path + "redo/" + str(j) + "_" + str(len(pixel_protocols)) + ".tif", both)

                center_RAM = [x, y, r]
                center_RAM_im = masked_im[:, 130:-130]
                found = True
                p2 = 25
                #dclass = 0#class_dict[part_meta[j][-4]]
                new_ims.append(center_RAM_im)  # segmented image
                d = res * center_RAM[2] * 2,
                RAM_labels.append([im_path, protocol, dclass])

            else:
                cv2.imwrite(write_path + "newly_skipped/" + str(j) + "_blank_" + str(len(pixel_protocols)) + ".tif", both)

                new_skipped.append(j)

            j += 1
            continue

        if crop == 'c':
            if len(pixel_protocols) > 1 and second_crop == True:
                print("pixel_protocols > 1", save_im.shape)
                # im = save_im
                # dx0, dx1 = int(((im.shape[1] - pixel_protocols[0]) // 2) - c_inc),  int(im.shape[1] - (im.shape[1] - pixel_protocols[0]) // 2 + c_inc)
                # dy0, dy1 = int(((im.shape[0] - pixel_protocols[0]) // 2) - c_inc), int(im.shape[0] - (im.shape[0] - pixel_protocols[0]) // 2 + c_inc)
            else:
                dx0, dx1 = int(((im.shape[1] - pixel_protocols[-1]) // 2) - c_inc), int(
                    im.shape[1] - (im.shape[1] - pixel_protocols[-1]) // 2 + c_inc)
                dy0, dy1 = int(((im.shape[0] - pixel_protocols[-1]) // 2) - c_inc), int(
                    im.shape[0] - (im.shape[0] - pixel_protocols[-1]) // 2 + c_inc)
            save_im = im
            mask0 = np.zeros_like(im)
            mask0[dy0:dy1, dx0:dx1] = im[dy0:dy1, dx0:dx1]
            im = mask0  # im[dy0:dy1, dx0:dx1]
            if (len(pixel_protocols) > 1) and second_crop == False:
                second_crop = True
                tried_crop = False
            else:
                second_crop = False
                tried_crop = True
        if p2 == 15:
            tried_p2 = True
        elif p2 == 95:
            tried_p2b = True

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        gray_blurred = cv2.blur(gray, (3, 3))
        detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 175, param1=35,
                                            param2=p2, minRadius=15, maxRadius=min(im.shape[0] // 2, im.shape[
                1] // 2))  # work for all other examples above

        # mark whether center RAM found and save set once found
        found = False
        center_RAM = [0, 0, 0]
        # RAMEO meta
        RAMEO_count, Rx, Ry = 0, 0, 0

        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles))
            for i in range(len(detected_circles[0, :])):
                mask = np.zeros_like(im)
                pt = detected_circles[0, i]
                a, b, r = pt[0], pt[1], pt[2]
                # black im
                mask = cv2.circle(mask, (a, b), r + r_inc, (255, 255, 255), -1)
                # thresh_im = cv2.circle(thresh_im, (a, b), r + r_inc + 5, (0, 0, 0), -1)
                masked_im = cv2.bitwise_and(im, mask)
                both = np.concatenate((im, masked_im), axis=1)

                x_dist = abs(im.shape[1] // 2 - a)
                y_dist = abs(im.shape[0] // 2 - b)
                r_low = 120
                r_high = 350
                if (((x_dist < from_center) and (y_dist < from_center)) and (r_high > r > r_low)):  # also check r
                    if found == True:
                        if (x_dist < x_min) and (y_dist <= y_min):
                            # ask user to change or not
                            min_x, min_y = x_dist, y_dist
                            center_RAM = [a, b, r]
                            center_RAM_im = masked_im[:, 130:-130]
                            center_RAM_im_both = both
                            found = True
                            p2 = 25
                    else:

                        if ((pixel_protocols[0] + p_inc) > (2 * r) > (pixel_protocols[0] - p_inc)):
                            print("")
                            # diffp = abs(pixel_protocols[0] - 2*r)
                            # print(diffp)
                        elif ((pixel_protocols[-1] + p_inc) > (2 * r) > (pixel_protocols[-1] - p_inc)):
                            if (len(pixel_protocols) > 1):
                                print("")
                                # pixel_protocols = pixel_protocols[-1]
                        else:
                            new_skipped.append(j)
                            diffp = abs(pixel_protocols[0] - 2 * r)
                            print("diff from protocol:", diffp)
                            if (len(pixel_protocols) > 1) and (abs(pixel_protocols[-1] - 2 * r) < diffp):
                                diffp = abs(pixel_protocols[-1] - 2 * r)
                            cv2.imwrite(write_path + "pp_off/" + str(j) + "_lenpp_" + str(len(pixel_protocols)) + "_diffp_" + str(diffp) + ".tif", both)
                            center_RAM_im = masked_im[:, 130:-130]
                            cv2.imwrite(write_path + "pp_off/" + str(j) + ".tif",center_RAM_im)
                            skipped.append(j)  # j += 1
                            continue

                        x_min, y_min = x_dist, y_dist
                        Rx, Ry = a, b
                        center_RAM = [a, b, r]
                        if debug_level == 'all':
                            print('RAM center, diameter in microns', a, b, 2 * r * res)
                            cv2.imshow("Detected CENTER Circles in: " + im_path, both)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()

                        found = True
                        center_RAM_im = masked_im[:, 130:-130]
                        #print("new im shape, center_RAM_im.shape, center_RAM:", center_RAM_im.shape, center_RAM)
                        cv2.imwrite(write_path + "good/" + str(j) + "_" + str(len(pixel_protocols)) + "_" + str(p2) + ".tif", both)
                        p2 = 25

                if (found == True) and i > 0:
                    # check distance between centers is less than r1 + r2. distance:
                    if math.sqrt((float(a) - center_RAM[0]) ** 2 + (float(b) - center_RAM[1]) ** 2) <= (
                            r + center_RAM[2]):
                        RAMEO_count += 1
                        RAMEO_labels.append([im_path, protocol, dclass])
                        if debug_level == 'RAMEOs':
                            cv2.imshow("Detected RAMEO Circles in: " + im_path, both)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                if (len(detected_circles[0, :]) - 1 == i):
                    if found == False:
                        if debug_level == 'undetected':
                            if tried_crop != True:
                                crop = 'c'
                                print('trying crop(s)')
                            elif tried_p2 != True:
                                p2 = 15
                                c = 'nc'
                                print('trying 15')
                            elif tried_p2b != True:
                                p2 = 95
                                print('trying 95')
                            else:
                                print("Skipping index ", j, ", and appending to newly skipped inds.")
                                new_skipped.append(j)
                                cv2.imwrite(write_path + "newly_skipped/" + str(j) + "_" + str(len(pixel_protocols)) + ".tif", im)
                                tried_crop = False
                                tried_p2 = False
                                tried_p2b = False
                                p2 = 25
                                j += 1

                    else:
                        #dclass = 0#class_dict[part_meta[j][-4]]
                        new_ims.append(center_RAM_im)  # segmented image
                        RAM_labels.append(
                            [im_path, protocol, dclass])


                        d = res * center_RAM[2] * 2


                        j += 1
        else:
            if debug_level == 'undetected':
                if tried_crop != True:
                    crop = 'c'
                    print('trying crop(s)')
                elif tried_p2 != True:
                    p2 = 15
                    c = 'nc'
                    print('trying 15')
                elif tried_p2b != True:
                    p2 = 95
                    print('trying 95')
                else:
                    print("Skipping index ", j, ", and appending to skipped inds.")
                    new_skipped.append(j)
                    cv2.imwrite(write_path + "newly_skipped/" + str(j) + "_" + str(
                        len(pixel_protocols)) + ".tif", im)
                    tried_crop = False
                    tried_p2 = False
                    tried_p2b = False
                    p2 = 25
                    j += 1

    with open(write_path + protocol + "_meta.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Server_Path", "Protocol"])
        writer.writerows(RAM_labels)
    print("RAM meta (rameo count, d, c_x, c_y) saved as 'RAM[protocol]-meta.csv, with length:", len(RAM_labels))


    ims_np = np.array(new_ims)
    np.save(write_path + 'extractedTest' + protocol + 's.npy', ims_np)
    print("Corresponding masked center RAMs saved as 'extractedRAMs.npy', check shape:", ims_np.shape)

    # print("notworkingp", notworkingp)
    print("Newly skipped", new_skipped)


if __name__ == "__main__":
    server = "/Volumes/"
    csv_path = "/Users/yancey5/Desktop/Projects/DRP/EJdata/715124_RAM_Classifications.csv" #server + "Microscopy/NIF_Metrology_Log.xlsm"
    write_in = "/Users/yancey5/Desktop/Projects/DRP/"


    protocols = ['RAM2000', "RAM1260", "RAM520", "RAM160", "RAM360,RAM600", "NA"]

    protocol = protocols[-1]
    write_path = write_in + protocol + "-ims" + "/"
    os.mkdir(write_path)

    debug_level = 'undetected'
    p2 = 25

    min_dist = 175
    r_inc = 8
    p_inc = 100
    c_inc = 100
    from_center = 50  # 100

    skipped = []

    damagedRAM_serverDataPrep(csv_path, server, write_path, protocol, skipped, debug_level, p2, min_dist, r_inc, p_inc, c_inc, from_center)
