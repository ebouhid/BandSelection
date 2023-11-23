nohup python svm.py allbands dataset_v4 > allbands.out 2>&1 &
nohup python svm.py pca pca_dataset > pca.out 2>&1 &
nohup python svm.py rgb rgb_dataset > rgb.out 2>&1 &
echo "Job submitted!"