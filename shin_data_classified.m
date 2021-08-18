clear all
close all
%% lengths of odorant tiffs
MO_len = 17;
CA_len = 15;
PB_len = 17;
EC_len = 19;
BA_len = 15;
BA_EC_len = 19;
PB_CA_len = 18;
glom_mask_len = 25;
num_samples = 15; %minimum length of tiff

%% select odorant combo and classification logic
select = "BA_EC"; %select either BA_EC or PB_CA odor combos
select_logic = "OR"; %AND, OR, XOR


%% selecting tiffs
if select == "BA_EC"
    select_0 = MO_len;
    select_1 = BA_len;
    select_2 = EC_len;
    select_3 = BA_EC_len;
    filename_list = ["MO.tif","BA.tif","EC.tif","BA+EC.tif"];
elseif select == "PB_CA"
    select_0 = MO_len;
    select_1 = PB_len;
    select_2 = CA_len;
    select_3 = PB_CA_len;
    filename_list = ["MO.tif","PB.tif","CA.tif","PB+CA.tif"];
end

%% importing tiffs and masks
total_len = select_0 + select_1 + select_2 + select_3;
total_len_list = [select_0,select_1,select_2,select_3];
len_after_list = [0,select_0, select_0+select_1, select_0+select_1+select_2,...
    select_0+select_1+select_2+select_3];
total_maps = zeros(256,256,total_len);
mask_fov = double(imread('mask_field-of-view.tif',1)/255);
target_maps = zeros(length(filename_list),total_len);
test_noise = rand(256*256,total_len)/25;
test_noisey = rand(256,256,total_len)/25;
glom_mask = zeros(256,256);

for i = 1:glom_mask_len
    glom_mask = glom_mask + double(imread('masks_all-glomeruli.tif',i))/255;
end
    
for i = 1:length(total_len_list)
    for j = 1:total_len_list(i)
        total_maps(:,:,len_after_list(i) + j) = ((imread(filename_list(i),j))).*mask_fov;
    end
    target_maps(i,len_after_list(i)+1:len_after_list(i+1)) = 1;
end


%% setting up targets
[OR_target, AND_target, XOR_target] = deal(ones(1,total_len));
OR_target(len_after_list(1)+1:len_after_list(2)) = 0;
AND_target(len_after_list(1)+1:len_after_list(4)) = 0;
XOR_target(len_after_list(1)+1:len_after_list(2)) = 0;
XOR_target(len_after_list(4)+1:len_after_list(5)) = 0;

%% shuffling order of tiff stacks and targets
total_maps_reshape = reshape(total_maps,[],total_len);
noisy_total_maps_reshape = total_maps_reshape + test_noise;

a = [randperm(total_len_list(1))+len_after_list(1),randperm(total_len_list(2))+len_after_list(2),...
    randperm(total_len_list(3))+len_after_list(3),randperm(total_len_list(4))+len_after_list(4)];

b = [randperm(total_len_list(1))+len_after_list(1),randperm(total_len_list(2))+len_after_list(2),...
    randperm(total_len_list(3))+len_after_list(3),randperm(total_len_list(4))+len_after_list(4)];

c = [randperm(total_len_list(1))+len_after_list(1),randperm(total_len_list(2))+len_after_list(2),...
    randperm(total_len_list(3))+len_after_list(3),randperm(total_len_list(4))+len_after_list(4)];


total_maps_reshape = total_maps_reshape(:,a);


OR_target = OR_target(a);
AND_target = AND_target(a);
XOR_target = XOR_target(a);

noisy_total_maps_reshape = noisy_total_maps_reshape(:,a);


%% plotting raw maps as images and in decision space

for i = 1:num_samples
    if i == 1 %plotting mean maps and mean location in decision space
        map_00 = mean(total_maps(:,:,len_after_list(1)+1:len_after_list(2)),3);
        map_10 = mean(total_maps(:,:,len_after_list(2)+1:len_after_list(3)),3);
        map_01 = mean(total_maps(:,:,len_after_list(3)+1:len_after_list(4)),3);
        map_11 = mean(total_maps(:,:,len_after_list(4)+1:len_after_list(5)),3);
        
        %% plotting raw mean maps
        figure('Position', [00 00 500 1000])
        clims = [0,.1];
        subplot(4,1,1)
        imagesc(map_00,clims)
        title('Neither odors presented (mineral oil)')
        subplot(4,1,2)
        imagesc(map_10,clims)
        title('First odor presented')
        subplot(4,1,3)
        imagesc(map_01,clims)
        title('Second odor presented')
        subplot(4,1,4)
        imagesc(map_11,clims)
        title('Both odors presented')
        
        %% calculating unit vectors in decision space and maps' projections along them
        v_10 = reshape(map_10,1,[]);
        u_10_raw = v_10; %first map is unit vector 1
        if u_10_raw == 0
            u_10 = u_10_raw;
        else
            u_10 = (u_10_raw)/norm(u_10_raw);
        end
        a_10 = dot(v_10,u_10);
        %v_10 = a_10*u_10;
        
        v_01 = reshape(map_01,1,[]);
        a_01 = dot(v_01,u_10);
        u_01_raw = v_01 - a_01*u_10; %second unit vector is second map minus its projection along first unit vector
        if u_01_raw == 0
            u_01 = u_01_raw;
        else
            u_01 = (u_01_raw)/norm(u_01_raw);
        end
        b_01 = dot(v_01,u_01);
        b_10 = dot(v_10,u_01);
        %v_01 = a_01*u_10 + b_01*u_01;
        
        v_11 = reshape(map_11,1,[]);
        a_11 = dot(v_11,u_10);
        b_11 = dot(v_11,u_01);
        u_11_raw = (v_11 - a_11*u_10 - b_11*u_01); %third unit vector is combined map minus its projection along first two unit vectors
        if u_11_raw == 0
            u_11 = u_11_raw;
        else
            u_11 = (u_11_raw)/norm(u_11_raw);
        end
        c_11 = dot(v_11,u_11);
        %v_11 = a_11*u_10 + b_11*u_01 + c_11*u_11;
        
        v_00 = reshape(map_00,1,[]);%reshape(o1_r_map,1,[]);
        a_00 = dot(v_00,u_10);
        b_00 = dot(v_00,u_01);
        c_00 = dot(v_00,u_11);
        
        
        
        %% vectors for plotting
        x = [a_00, a_10, a_01, a_11];
        y = [b_00, b_10, b_01, b_11];
        z = [c_00, 0, 0, c_11];
        
        maxxy = max([x y z]);
        
        %% plotting means in decision space
        figure('Position', [950 00 1000 1000])
        scatter3(x(1),y(1),z(1),500,'b','filled')
        title('1 map decision space')
        hold on
        scatter3(x(2),y(2),z(2),500,'r','filled')
        title('1 map decision space')
        hold on
        scatter3(x(3),y(3),z(3),500,'g','filled')
        title('1 map decision space')
        hold on
        scatter3(x(4),y(4),z(4),500,'k','filled')
        title('1 map decision space')
        legend('map 00','map 10','map 01','map 11','AutoUpdate','off')
        hold on
    else %plotting individual maps in decision space (unit vectors already calculated above)
        map_00 = total_maps(:,:,len_after_list(1)+i);
        map_10 = total_maps(:,:,len_after_list(2)+i);
        map_01 = total_maps(:,:,len_after_list(3)+i);
        map_11 = total_maps(:,:,len_after_list(4)+i);
        
        v_00 = reshape(map_00,1,[]);
        a_00 = dot(v_00,u_10);
        b_00 = dot(v_00,u_01);
        c_00 = dot(v_00,u_11);
        
        v_10 = reshape(map_10,1,[]);
        a_10 = dot(v_10,u_10);
        b_10 = dot(v_10,u_01);
        c_10 = dot(v_10,u_11);
        
        v_01 = reshape(map_01,1,[]);
        a_01 = dot(v_01,u_10);
        b_01 = dot(v_01,u_01);
        c_01 = dot(v_01,u_11);
        
        v_11 = reshape(map_11,1,[]);
        a_11 = dot(v_11,u_10);
        b_11 = dot(v_11,u_01);
        c_11 = dot(v_11,u_11);
        
        
        
        x = [a_00, a_10, a_01, a_11];
        x2(i-1,:) = x;
        y = [b_00, b_10, b_01, b_11];
        y2(i-1,:) = y;
        z = [c_00, c_10, c_01, c_11];
        z2(i-1,:) = z;
        
        
       
         

        scatter3(x(1),y(1),z(1),500,'b*')
        title('1 map decision space')
        hold on
        scatter3(x(2),y(2),z(2),500,'r*')
        title('1 map decision space')
        hold on
        scatter3(x(3),y(3),z(3),500,'g*')
        title('1 map decision space')
        hold on
        scatter3(x(4),y(4),z(4),500,'k*')
        title('1 map decision space')
        if i == num_samples
            maxxy = max(max([x2 y2 z2]));
            axis([-1 maxxy+1 -1 maxxy+1 -1 maxxy+1])
        end
        hold on
    end
end
axis manual 

%% perceptron classifier
if select_logic == "AND"
    target = AND_target;
elseif select_logic == "OR"
    target = OR_target;
elseif select_logic == "XOR"
    target = XOR_target;
end
    
net = perceptron;

net.trainParam.max_fail = 100;
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.6;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0.2;


[net, tr_a] = train(net,total_maps_reshape,target);
predicted = net(total_maps_reshape(:,tr_a.testInd));
perc_correct = 1 - sum(abs(predicted-target(tr_a.testInd)))/length(tr_a.testInd);

%% plot weight vector and orthogonal plane
figure(2)
weights = net.iw{1,1};
we1 = dot(weights,u_10);
we2 = dot(weights,u_01);
we3 = dot(weights,u_11);

quiver3(0,0,0,we1,we2,we3);


v = [we1 we2 we3];

we0 = abs(net.b{1,1});
int = we0/we3;

x_1 = we1*we0/(we1^2 + we2^2 + we3^2);
x_2 = we2*we0/(we1^2 + we2^2 + we3^2);
x_3 = we3*we0/(we1^2 + we2^2 + we3^2);

w = null(v); % Find two orthonormal vectors which are orthogonal to v
[P,Q] = meshgrid(-8:8); % Provide a gridwork (you choose the size)
X = x_1+w(1,1)*P+w(1,2)*Q; % Compute the corresponding cartesian coordinates
Y = x_2+w(2,1)*P+w(2,2)*Q; %   using the two vectors in w
Z = x_3+w(3,1)*P+w(3,2)*Q;
surf(X,Y,Z,'FaceAlpha',0.8)

%% report single layer classifier accuracy
sprintf('Percent correct for %s %.2f', select_logic,100*perc_correct)