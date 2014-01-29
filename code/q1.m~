function q1()
cd ../data
training_data = load('train_small.mat');
test_data = load('test.mat');
cd ../code/liblinear-1.94/matlab

X = zeros(1,7);
Y = zeros(1,7);

test_images=test_data.test.images;
test_labels=test_data.test.labels;
test_mtx = zeros(size(test_images,3), size(test_images,1)*size(test_images,2));
for r=1:size(test_images,1),
    for s=1:size(test_images,2),
	for t=1:size(test_images,3),
	    test_mtx(t,size(test_images,1)*(r-1)+s) = test_images(r,s,t);
        end
    end
end

for k=1:size(training_data.train,2),
    images=training_data.train{k}.images;
    labels=training_data.train{k}.labels;
    mtx = zeros(size(images,3),size(images,1)*size(images,2));
    for x=1:size(images,1),
        for y=1:size(images,2),
            for z=1:size(images,3),
                mtx(z,size(images,1)*(x-1)+y) = images(x,y,z);
            end
        end
    end
    model = train(labels,sparse(mtx), '-c 0.0000002');
    predicted_labels = predict(test_labels, sparse(test_mtx), model);
    cd ../..
    X(k) = size(images,3);
    Y(k) = benchmark(predicted_labels, test_labels);
    cd liblinear-1.94/matlab
end
cd ../..
plot(X,Y)
