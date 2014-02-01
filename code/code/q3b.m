function q3b()

rng(0);

cd ../data
training_data = load('train_small.mat');
test_data = load('test.mat');
cd ../code/liblinear-1.94/matlab

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

images=training_data.train{7}.images;
labels=training_data.train{7}.labels;
train_mtx = zeros(size(images,3)*9/10,size(images,1)*size(images,2));
valid_mtx = zeros(size(images,3)*1/10,size(images,1)*size(images,2));
train_labels = zeros(size(images,3)*9/10,1);
valid_labels = zeros(size(images,3)*1/10,1);

error = zeros(1,10);

l_bound = -7.5;
u_bound = -6.5;
% step_size cannot be 1.
step_size = 8;
iterations = 2;

for iter=1:iterations;
    best_pten = l_bound;
    best_error = 1;
    for p_ten=linspace(l_bound, u_bound, step_size),
        v = randperm(size(training_data.train{7}.labels, 1));
        c = 10^(p_ten);
        for k=1:10,
            valid_indices = v((k-1)*1000+1:k*1000);
            training_indices = [v(1:(k-1)*1000), v(k*1000+1:10000)];
            train_num = 1;
            valid_num = 1;

            for ti = training_indices,
                for x=1:size(images,1),
                    for y=1:size(images,2),
                        train_mtx(train_num,size(images,1)*(x-1)+y) = images(x,y,ti);
                        train_labels(train_num) = labels(ti);
                    end
                end
                train_num = train_num + 1;
            end
            for vi = valid_indices,
                for x=1:size(images,1),
                    for y=1:size(images,2),
                        valid_mtx(valid_num,size(images,1)*(x-1)+y) = images(x,y,vi);
                        valid_labels(valid_num) = labels(vi);
                    end
                end
                valid_num = valid_num + 1;
            end

            model = train(train_labels, sparse(train_mtx), ['-c ' num2str(c) ' -q']);
            predicted_labels = predict(valid_labels, sparse(valid_mtx), model);
            cd ../..
            error(k) = benchmark(valid_labels, predicted_labels);
            cd liblinear-1.94/matlab
        end
        c
        avg_error = mean(error)
        if avg_error < best_error
            best_error = avg_error;
            best_pten = p_ten;
        end
        arr(k) = avg_error;
    end
    step = (u_bound - l_bound)./(step_size-1);
    l_bound = best_pten - step;
    u_bound = best_pten + step;
end

bestc = 10^(best_pten)
mtx = zeros(size(images,3),size(images,1)*size(images,2));
for x=1:size(images,1),
    for y=1:size(images,2),
        for z=1:size(images,3),
            mtx(z,size(images,1)*(x-1)+y) = images(x,y,z);
        end
    end
end
model = train(labels,sparse(mtx), ['-q -c ' num2str(bestc)]);
predicted_labels = predict(test_labels, sparse(test_mtx), model);
cd ../..
benchmark(predicted_labels, test_labels)

