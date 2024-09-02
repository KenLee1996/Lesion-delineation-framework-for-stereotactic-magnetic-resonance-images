function NameOfROI=RTSSimport(rtsspath,niftiimgpath)
% if isempty(dir([niftiimgpath '\tv*']))==1
% niftiimg=niftiread([niftiimgpath '\img.nii.gz']);
% niftiimginfo=niftiinfo([niftiimgpath '\img.nii.gz']);
niftiimg=niftiread([niftiimgpath '\img.nii']);
niftiimginfo=niftiinfo([niftiimgpath '\img.nii']);
if exist([rtsspath '\RTSS.dcm'],'file')
    dicom=dicominfo([rtsspath '\RTSS.dcm'],'UseDictionaryVR',true);
elseif exist([rtsspath '\rtss.dcm'],'file')
    dicom=dicominfo([rtsspath '\rtss.dcm'],'UseDictionaryVR',true);
elseif exist([rtsspath '\RTSS.dcm_d.dcm'],'file')
    dicom=dicominfo([rtsspath '\RTSS.dcm_d.dcm'],'UseDictionaryVR',true);
else
    rtssfilename=dir([rtsspath '\RS*']);
    dicom=dicominfo([rtsspath '\' rtssfilename(1).name],'UseDictionaryVR',true);
end
nROIss=numel(fieldnames(dicom.StructureSetROISequence));
for n=1:nROIss;nnR{n}=['Item_' num2str(n)];end
NameOfROI={};
for j=1:nROIss
    nn=[];
    if isfield(dicom.ROIContourSequence.(nnR{j}),'ContourSequence')
        nROI=numel(fieldnames(dicom.ROIContourSequence.(nnR{j}).ContourSequence));        
        for n=1:nROI;nn{n}=['Item_' num2str(n)];end
        
        % if ~contains(lower(dicom.StructureSetROISequence.(nnR{j}).ROIName),'skull')
            NameOfROI{j,1}=dicom.StructureSetROISequence.(nnR{j}).ROIName;
            dicom_img_file=dir([rtsspath '\*.dcm']);
            tvimg=zeros(size(niftiimg));
            
            slice_location=[];
            for k=1:length(dicom_img_file)
                if ~contains(lower(dicom_img_file(k).name),'rt') && ~contains(lower(dicom_img_file(k).name),'rs')
                    tmp_info=dicominfo([rtsspath '\' dicom_img_file(k).name],'UseDictionaryVR',true);
%                     if isfield(tmp_info,'SliceLocation')
%                         slice_location=[slice_location tmp_info.SliceLocation];
%                     else
                        slice_location=[slice_location tmp_info.ImagePositionPatient(3)];
%                     end
                end
            end
            slice_location=sort(slice_location);
            
            for i=1:nROI
                sln=[];
                ncp=dicom.ROIContourSequence.(nnR{j}).ContourSequence.(nn{i}).NumberOfContourPoints;
                contour=dicom.ROIContourSequence.(nnR{j}).ContourSequence.(nn{i}).ContourData;
                contour=reshape(contour,[3,ncp])';
                RefUID=dicom.ROIContourSequence.(nnR{j}).ContourSequence.(nn{i}).ContourImageSequence.Item_1.ReferencedSOPInstanceUID;
                for k=1:length(dicom_img_file)
                    if ~contains(lower(dicom_img_file(k).name),'rtss')
                        tmp_info=dicominfo([rtsspath '\' dicom_img_file(k).name],'UseDictionaryVR',true);
                        % [~,tmp_name,~]=fileparts([rtsspath '\' dicom_img_file(k).name]);               
                        if strcmp(tmp_info.SOPInstanceUID,RefUID)==1
                            img=dicomread([rtsspath '\' dicom_img_file(k).name]);
                            info=dicominfo([rtsspath '\' dicom_img_file(k).name],'UseDictionaryVR',true);
                            %                             if isfield(info,'SliceLocation')
                            %                                 sln=find(slice_location==info.SliceLocation);
                            %                             else
                            sln=find(slice_location==info.ImagePositionPatient(3));
                            %                             end
                            break;
                        end
                    end
                end
                % if ~isempty(sln)
                    ncontour=zeros(size(contour,1),size(contour,2),size(contour,3));
                    % affine=zeros(3,3);
                    % affine(:,3)=info.ImagePositionPatient(:);
                    % affine(:,1)=info.ImageOrientationPatient(1:3);
                    % affine(:,2)=info.ImageOrientationPatient(4:6);
                    % affine(:,1:2)=affine(:,1:2)*info.PixelSpacing(1);
                    affine=zeros(4,4);
                    affine(1:3,4)=info.ImagePositionPatient(:);
                    affine(4,4)=1;
                    affine(1:3,1)=info.ImageOrientationPatient(1:3);
                    affine(1:3,2)=info.ImageOrientationPatient(4:6);
                    affine(1:3,1:2)=affine(1:3,1:2)*info.PixelSpacing(1);
                    for con_ind=1:size(ncontour,1)
                        tmp_c=round(pinv(affine)*[contour(con_ind,1);contour(con_ind,2);contour(con_ind,3);1]);
                        ncontour(con_ind,1:2)=tmp_c(1:2);
                        ncontour(con_ind,3)=info.ImagePositionPatient(3);
                    end

                    if size(niftiimg,2)>size(niftiimg,3)
                        BW = roipoly(img,ncontour(:,1),ncontour(:,2));
                    else
                        BW = roipoly(img,ncontour(:,1),abs(ncontour(:,3)));
                    end
                    tvimg(:,:,sln)=tvimg(:,:,sln)+double(rot90(BW,3));
                % end
            end
            tvimg(tvimg>0)=1;
            tvimg=int16(tvimg);
            niftiimginfo.Datatype='int16';
            if niftiimginfo.Transform.T(1,1)>0
                tvimg=flipud(tvimg);
            end
            if niftiimginfo.Transform.T(2,2)<0
                tvimg=fliplr(tvimg);
            end
            if contains(NameOfROI{j,1},'*')
                NameOfROI{j,1}=replace(NameOfROI{j,1},'*','');                
            end
            fprintf('%s\n',NameOfROI{j,1});
            niftiwrite(tvimg,[niftiimgpath '\' NameOfROI{j,1} '.nii'],niftiimginfo,'Compressed',true);
        % end
    end
end
% end