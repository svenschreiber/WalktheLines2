"""
Code adapted from Max Braeschke's bachelor thesis:
    'Tracking and Closing Contours of Various Objects through Walk the Lines 2'
"""
from matplotlib.image import imsave
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import sobel_v, sobel_h, gaussian
from skimage.measure import label
import os
import numpy as np
from skimage.morphology import diamond, opening, erosion, binary_closing, skeletonize
from argparse import ArgumentParser
from pathlib import Path

def getCropIndizes(startx, starty, angle, cropLength):
    # Generate a list of tuples representing the indices of the cropped region
    tupleList = [(starty, startx)]
    
    # Iterate over a range of values within [-cropLength, cropLength)
    for i in range(-cropLength, cropLength):
        # Calculate new coordinates based on angle
        newy = starty + i * np.sin(np.radians(angle))
        newx = startx + i * np.cos(np.radians(angle))
        
        # Round the coordinates and append to the tuple list
        tupleList.append((round(newy), round(newx)))
    
    return tupleList

def cutIsValidByZeros(startx, starty, angle, cropLength, img, bone, btwo, labeledImg):
    try:
        # Calculate coordinates for the end of the crop
        newy = starty + cropLength * np.sin(np.radians(angle))
        newx = startx + cropLength * np.cos(np.radians(angle))
        
        # Check if the end of the crop has zeros in the image
        hasZerosEnd = img[round(newy), round(newx)] == 0
        
        # Get the label of the first pixel in the cropped region
        labeledFirst = labeledImg[round(newy), round(newx)]
        
        # Check if the first pixel is on the expected label spaces
        isOnGoodSpace = (labeledFirst == bone) or (labeledFirst == btwo)
        
        # Return False if any of the conditions is not met
        if not hasZerosEnd or not isOnGoodSpace:
            return False
        
        # Calculate coordinates for the other end of the crop
        newy = starty + -cropLength * np.sin(np.radians(angle))
        newx = startx + -cropLength * np.cos(np.radians(angle))
        
        # Check if the other end of the crop has zeros in the image
        hasZerosEnd = img[round(newy), round(newx)] == 0
        
        # Get the label of the second pixel in the cropped region
        labeledSecond = labeledImg[round(newy), round(newx)]
        
        # Check if the second pixel is on the expected label spaces and is different from the first pixel
        isOnGoodSpace = (labeledSecond == bone) or (labeledSecond == btwo)
        isNotSameSpace = labeledSecond != labeledFirst
        
        # Return False if any of the conditions is not met
        if not hasZerosEnd or not isOnGoodSpace or not isNotSameSpace:
            return False
        
        foundFirstNumbers = False
        foundZeroAfterNumbers = False
        
        # Iterate over the range of values within [-cropLength+1, cropLength-1)
        # Do that to check wether the cut is connecting more then to background areas
        for i in range(-cropLength + 1, cropLength - 1):
            # Calculate coordinates for each pixel within the cropped region
            newy = starty + i * np.sin(np.radians(angle))
            newx = startx + i * np.cos(np.radians(angle))
            
            # Check if the pixel is a number or zero
            isNumber = img[round(newy), round(newx)] != 0
            
            # Update flags based on the encountered pixels
            if isNumber and not foundFirstNumbers:
                foundFirstNumbers = True
            if not isNumber and foundFirstNumbers and not foundZeroAfterNumbers:
                foundZeroAfterNumbers = True
            if isNumber and foundFirstNumbers and foundZeroAfterNumbers:
                return False
        
        return True
    except:
        return False

def cut_at_best(img, cut_range, plot_cut=False):
    # Create a copy of the input image
    img = img.copy()

    # Calculate the orientation of edges in the image
    img_smoothed = gaussian(img, 5)
    orientation = np.degrees(np.atan(sobel_h(img_smoothed)/(1e-10+sobel_v(img_smoothed))))

    # Convert the image to a binary image where non-zero values are set to 1
    img_binary = np.where(img > 0, 1, 0)

    # Label the connected components in the binary image
    img_labeled = label(img_binary, background=1, connectivity=1)

    # Count the occurrences of each label in the labeled image
    bins = -np.bincount(img_labeled.flatten())
    bins[0] = 0
    bins = np.argsort(bins)

    # Sort the image pixels in descending order based on their intensity
    coords = np.unravel_index(np.argsort(img, axis=None), img.shape)
    coords = np.array(coords).T[::-1]

    for py, px in coords:
        p_orient = orientation[py][px]

        # Check if the cut is valid based on zeros, labels, and orientation. If not valid, continue to the next pixel.
        if not cutIsValidByZeros(px, py, p_orient, cut_range, img, bins[0], bins[1], img_labeled):
            continue

        dir_y, dir_x = np.sin(np.radians(p_orient)), np.cos(np.radians(p_orient))
        newy = py + cut_range * dir_y
        newx = px + cut_range * dir_x

        newDownY = py + -cut_range * dir_y
        newDownX = px + -cut_range * dir_x

        cutIndizes = getCropIndizes(px, py, p_orient, cut_range)
        leftFromCut = (round(py + np.sin(np.radians(p_orient + 90))), round(px + np.cos(np.radians(p_orient + 90))))
        rightFromCut = (round(py + np.sin(np.radians(p_orient - 90))), round(px + np.cos(np.radians(p_orient - 90))))

        newImg = img.copy()

        # Set the pixels within the cropped region to zero in the new image
        for idx in cutIndizes:
            newImg[idx[0]][idx[1]] = 0

        # Adjust the left neighbor of the cut if necessary
        if abs(py - leftFromCut[0]) == 1 and abs(px - leftFromCut[1]) == 1:
            lfcnewy = leftFromCut[0] + (py - leftFromCut[0])
            leftFromCut = (lfcnewy, leftFromCut[1])

        # Get the indices of the second cropped region
        secondCutIndizes = getCropIndizes(leftFromCut[1], leftFromCut[0], p_orient, cut_range)
        leftFromCut = (round(leftFromCut[0] + np.sin(np.radians(p_orient + 90))), round(leftFromCut[1] + np.cos(np.radians(p_orient + 90))))

        # Set the pixels within the second cropped region to zero in the new image
        for idx in secondCutIndizes:
            newImg[idx[0], idx[1]] = 0

        # Set the pixels between the two cropped regions to zero in the new image
        for idx, idxLeft in zip(cutIndizes, secondCutIndizes):
            newImg[idx[0]:idxLeft[0], idx[1]:idxLeft[1]] = 0

        # Check if the pixels at the left and right neighbors of the cut have the same intensity. If not, continue to the next pixel.
        if newImg[leftFromCut[0], leftFromCut[1]] != img[py, px] or newImg[rightFromCut[0], rightFromCut[1]] != img[py, px]:
            continue

        th = np.max(img)

        buildImg = np.where(newImg == th, newImg, 0)

        # Label the connected components in the binary image
        buildImg = label(buildImg, background=0, connectivity=2)

        # Adjust the threshold until the left and right neighbors are connected
        while (
            (buildImg[leftFromCut[0], leftFromCut[1]] != buildImg[rightFromCut[0], rightFromCut[1]])
            or (buildImg[leftFromCut[0], leftFromCut[1]] == 0 and 0 == buildImg[rightFromCut[0], rightFromCut[1]])
        ):
            th = th - 0.5
            buildImg = np.where(newImg >= th, 1, 0)
            buildImg = label(buildImg, background=0, connectivity=2)
                
        print(th)
        buildImg = np.where(newImg>=th, 1, 0)
        buildImg = label(buildImg, background=0, connectivity=2)

        # Plotting
        if plot_cut:
            plt.figure("Cut")
            plt.imshow(newImg, cmap="gray", vmin=0, vmax=np.max(newImg))
            plt.plot(px, py, marker= "v", color = "blue")
            plt.plot((newDownX, newx), (newDownY, newy), color="blue", linewidth=3)
            plt.plot(leftFromCut[1], leftFromCut[0], marker= ".", color = "red")
            plt.plot(rightFromCut[1], rightFromCut[0], marker= ".", color = "red")
        
        # Check if the threshold is less than or equal to 10 or if the pixel at the right neighbor is zero. If so, continue to the next pixel.
        if th <= 10 or buildImg[rightFromCut[0], rightFromCut[1]] == 0: continue

        orgImgConnected = np.where(newImg>=th, img, 0)
        orgImgConnected = np.where((th < orgImgConnected) & (orgImgConnected <= th + 0.5), np.max(img), orgImgConnected)

        buildImg = orgImgConnected

        buildImg = np.where(buildImg > 0, 1, 0)
        buildImg = label(buildImg, background= 0 , connectivity=2)
        binCount = np.bincount(buildImg.flatten())
        binCount[0] = 0 
        freqValue = np.argmax(binCount)
        buildImg = np.where(buildImg == freqValue, 1, 0)

        for idx in cutIndizes:
            if(img[idx[0], idx[1]] >= th):
                buildImg[idx[0], idx[1]] = 1

        for idx in secondCutIndizes:
            if(img[idx[0], idx[1]] >= th):
                buildImg[idx[0], idx[1]] = 1

        break

    return buildImg

def get_largest_component(mask, background=1, inverse=False):
    lb = label(mask, background=background, connectivity=1)
    hI = np.argmax(np.bincount(lb.flatten()))
    return np.where(lb == hI, 1, 0) if not inverse else np.where(lb == hI, 0, 1)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("src_dir")
    parser.add_argument("rgb_dir")
    parser.add_argument("out_dir")
    parser.add_argument('--mode', choices=['eros', 'thinned'], help='execution mode', default='eros')
    parser.add_argument('--visualize', action="store_true", help="plot results")
    args = parser.parse_args()

    for filename in os.listdir(args.src_dir):
        if not filename.endswith(('.jpg', '.png', 'jpeg')):
            continue

        print(filename)

        # Read the image file
        img = imread(os.path.join(args.src_dir, filename))
        
        # Convert image to grayscale if it has multiple channels
        if len(img.shape) >= 3: img = img[:, :, 0]

        # Check if the most frequent pixel value is 255 (white)
        most_frequent = np.argmax(np.bincount(img.flatten()))
        if most_frequent == 255:
            # Invert the image if it's mostly white
            img = np.abs(img.max() - img)

        # Crop the image by removing a 10-pixel border
        border_size = 10
        img = img[border_size:-border_size, border_size:-border_size]

        # Set the last row and column of the image to zero
        height, width = img.shape
        img[height - 1, :] = 0
        img[:, width - 1]  = 0

        # Perform the "cut_at_best" method on the image with a cut size of 12 pixels
        buildImg = cut_at_best(img, cut_range=12, plot_cut=args.visualize)
        
        # Check if the resulting image contains only zeros and ones
        if not np.all(np.isin(buildImg.flatten(), (0, 1))):

            # Convert the image to binary and label the connected components
            buildImg = np.where(buildImg > 0, 1, 0)
            buildImg = label(buildImg, background=0, connectivity=2)

            # Find the most frequent label value
            binCount = np.bincount(buildImg.flatten())
            binCount[0] = 0
            freqValue = np.argmax(binCount)
            buildImg = np.where(buildImg == freqValue, 1, 0)

        # Create a diamond-shaped filter
        filter = diamond(1)

        outMask = get_largest_component(buildImg, background=1)

        if args.mode == "eros":
            # Invert the mask and apply morphological opening
            opened = opening(1 - outMask, footprint=filter)
            
            # Get the largest component again after opening
            out_mask_morphed = get_largest_component(opened, background=0, inverse=True)

        elif args.mode == "thinned":
            # Skeletonize and close the original image
            thinned = binary_closing(skeletonize(buildImg))
            
            # Invert after extracting the largest component from thinned
            out_mask_morphed = get_largest_component(thinned, background=1, inverse=True)

        # Generate the final contour by subtracting erosion from the opened mask
        finalContour = out_mask_morphed - erosion(out_mask_morphed, footprint=filter)

        # Determine the corresponding original image filename
        if os.path.exists(args.rgb_dir + "/" + filename[0: filename.rindex(".")] + ".JPEG"):
            orgImg = imread(args.rgb_dir + "/" + filename[0: filename.rindex(".")] + ".JPEG")
        else:
            orgImg = imread(args.rgb_dir + "/" + filename[0: filename.rindex(".")] + ".jpg")

        # Crop the masks and filtered image by removing a 40-pixel border
        outMask = outMask[40:-40, 40:-40]
        out_mask_morphed = out_mask_morphed[40:-40, 40:-40]

        # Invert the mask
        outMask = np.where(outMask == 1, 0, 1)
        orgImgFiltered = np.zeros_like(orgImg)
        
        # Copy pixels from the original image to the filtered image based on the inverted mask
        for idy in range(0, out_mask_morphed.shape[0] - 1):
            for idx in range(0, out_mask_morphed.shape[1] - 1):
                if out_mask_morphed[idy, idx] == 1:
                    orgImgFiltered[idy, idx] = orgImg[idy, idx]

        if args.visualize:
            # Display the contour image
            plt.figure("Contour")
            plt.imshow(finalContour)

            # Display the segmented image
            plt.figure("Segmentation")
            plt.imshow(orgImgFiltered)

            # Display the binary mask image
            plt.figure("BinaryMask")
            plt.imshow(out_mask_morphed)

        filename_no_ext = Path(filename).stem
        output_dir = os.path.join(args.out_dir, args.mode)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        imsave(os.path.join(output_dir, f"{filename_no_ext}_segmentation.png"), orgImgFiltered)
        imsave(os.path.join(output_dir, f"{filename_no_ext}_binary_contour.png"), finalContour, cmap="gray", vmin=0, vmax=1)
        imsave(os.path.join(output_dir, f"{filename_no_ext}_binary_mask.png"), out_mask_morphed, cmap="gray", vmin=0, vmax=1)
        
        # Show the plotted images
        plt.show()