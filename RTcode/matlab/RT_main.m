% scan RTSS, RTDOSE and image dicom files
% convert the corresponding image dicom files (RT simulation CT or MR images)
% export the RT dose volumes and align to image volume
% export the ROIs from RTSS
%% read RT doss and overlay on DICOM
% base_path is the directory that contains all the dicom files
base_path='D:\2024_榮台聯大\AVM_data\TVGH\AVM\test\251936-0';
[imagefiles, rtssfiles, dosefiles] = ScanDICOMpath(base_path);
%% move image dicom files to isolated folder
if exist(fullfile(base_path,'dicom'),'dir')==0
    mkdir(fullfile(base_path,'dicom'));
    for dicom_file_ind=1:length(imagefiles)
        movefile([base_path '\' imagefiles{dicom_file_ind}], ...
            [base_path '\dicom']);
    end
    fprintf('%s\n','move image dicom files to dicom folder');
end
%% convert image dicom files to single nifti file
Option='-b n -z n -f img';
if isempty(dir([base_path '\dicom\*.nii']))
    eval(['!dcm2niix.exe ' Option ' ' base_path '\dicom']);
    movefile([base_path '\dicom\img.nii'], ...
        base_path);
end
%% move RT files to dicom folder
rtdicom_file_list=dir([base_path '\*.dcm']);
for rtdicom_file_ind=1:length(rtdicom_file_list)
    movefile([base_path '\' rtdicom_file_list(rtdicom_file_ind).name], ...
        [base_path '\dicom']);
end
fprintf('%s\n','move RT dicom files to dicom folder');
%% read and convert the RTDOSE dicom files to dose volume
RTDOSE_export_volume(base_path,dosefiles,imagefiles);
fprintf('%s\n','convert RTDOSE dicom files to dose volume');
%% read and convert the ROIs inside the RTSS dicom files to nifti volume
rtsspath=[base_path '\dicom'];
niftiimgpath=base_path;
NameOfRTSSROI=RTSSimport(rtsspath,niftiimgpath);
fprintf('%s\n','convert RTSS ROIs to the volume');
%% 