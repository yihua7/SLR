function [] = video_to_jpg(video_path,obj_path)
%% Convert mp4 to jpg
%% Read video   
video_obj=VideoReader(video_path);   

frame_number=video_obj.NumberOfFrames;

%% Get image frame by frame
if ~exist(obj_path,'dir')
    mkdir(obj_path);
    disp('successfully create directory!');
end

for i=1:frame_number
    image_name=[obj_path, '/', sprintf('%04d', i) '.jpg'];
    frame=read(video_obj,i);
    imwrite(frame,image_name, 'jpg');
end

disp('all images are written into directory image')
end

