function RTDOSE_export_volume(base_path,dosefiles,imagefiles)
%% load RT dose
for dosefile_ind=1:length(dosefiles)
    % dosefile_ind=3;
    dose_info=dicominfo(fullfile(base_path,'dicom',dosefiles{dosefile_ind}));
    raw_dose_data=squeeze(dicomread(dose_info));
    dose_data=double(raw_dose_data)*dose_info.DoseGridScaling;
    if dose_info.GridFrameOffsetVector(end) < dose_info.GridFrameOffsetVector(1)
        dose_data=flip(dose_data,3);
    end
    dose_data_slicethickness=abs(mean(diff(dose_info.GridFrameOffsetVector)));
    dose_resolution=[dose_info.PixelSpacing;dose_data_slicethickness];
    dose_origin=dose_info.ImagePositionPatient;
    dose_origin(3)=min(dose_info.ImagePositionPatient(3)+dose_info.GridFrameOffsetVector);

    %% load DICOM images
    slice_location=[];
    for dicom_ind=1:length(imagefiles)
        tmp_info=dicominfo(fullfile(base_path,'dicom',imagefiles{dicom_ind}));
        slice_location(length(slice_location)+1)=tmp_info.ImagePositionPatient(3);
    end
    [~,slice_index]=sort(slice_location,'ascend');
    tmp_info=dicominfo(fullfile(base_path,'dicom',imagefiles{slice_index(1)}));
    image_resolution=[tmp_info.PixelSpacing;tmp_info.SliceThickness];
    image_origin=tmp_info.ImagePositionPatient;


    image_data=zeros(tmp_info.Height,tmp_info.Width,length(imagefiles));
    for ind=1:length(slice_index)
        tmp_info=dicominfo(fullfile(base_path,'dicom',imagefiles{slice_index(ind)}));
        tmp_data=dicomread(tmp_info);
        image_data(:,:,ind)=double(tmp_data);
    end
    %% create grid base on origin points of dose and image
    [tarX, tarY, tarZ]=meshgrid(dose_origin(1):dose_resolution(1):dose_origin(1)+dose_resolution(1)*(size(dose_data,2)-1), ...
        dose_origin(2):dose_resolution(2):dose_origin(2)+dose_resolution(2)*(size(dose_data,1)-1), ...
        dose_origin(3):dose_resolution(3):dose_origin(3)+dose_resolution(3)*(size(dose_data,3)-1));

    [refX, refY, refZ]=meshgrid(image_origin(1):image_resolution(1):image_origin(1)+image_resolution(1)*(size(image_data,2)-1), ...
        image_origin(2):image_resolution(2):image_origin(2)+image_resolution(2)*(size(image_data,1)-1), ...
        image_origin(3):image_resolution(3):image_origin(3)+image_resolution(3)*(size(image_data,3)-1));
    %% interpolate the new dose data
    new_dose_data = interp3(tarX, tarY, tarZ, ...
        dose_data, ...
        refX,refY, refZ, '*linear', 0);
    %% display
    % figure, colormap([gray(255); jet(64)])
    % tmpimage=image_data;
    % tmpimage=(tmpimage-min(tmpimage(:)))/(max(tmpimage(:))-min(tmpimage(:)))*255;
    % tmpdose=new_dose_data;
    % tmpdose=(tmpdose-min(tmpdose(:)))/(max(tmpdose(:))-min(tmpdose(:)))*64+255;
    % for i=1:size(image_data,3)
    %     image(tmpimage(:,:,i)), hold on
    %     image(tmpdose(:,:,i),'alphadata',(new_dose_data(:,:,i)~=0)*0.3)
    %     title(['Slice #' num2str(i)]),axis off
    %     pause
    % end
    %% save as nifti file
    img_nii_info=niftiinfo(fullfile(base_path,'img.nii'));
    nifti_dose_data=single(rot90(new_dose_data,-1));
    img_nii_info.Datatype='single';
    [~,filename,~]=fileparts(dosefiles{dosefile_ind});
    niftiwrite(nifti_dose_data,fullfile(base_path,[filename '.nii']),img_nii_info,'Compressed',true);
end