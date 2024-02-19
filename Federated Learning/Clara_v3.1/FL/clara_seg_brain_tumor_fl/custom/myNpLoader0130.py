import numpy as np
import logging
from skimage.transform import resize
from scipy import ndimage

from ai4med.common.constants import ImageProperty
from ai4med.common.medical_image import MedicalImage
from ai4med.common.shape_format import ShapeFormat
from ai4med.common.transform_ctx import TransformContext
from ai4med.utils.dtype_utils import str_to_dtype
from ai4med.components.transforms.multi_field_transformer import MultiFieldTransformer

class MyNumpyLoader(MultiFieldTransformer):
    """Load Image from Numpy files.
    Args:
        shape (ShapeFormat): Shape of output image.
        dtype : Type for output data.
    """
    def __init__(self, fields, shape, dtype="float32"):
        MultiFieldTransformer.__init__(self, fields=fields)
        self._dtype = str_to_dtype(dtype)
        self._shape = ShapeFormat(shape)
        self._reader = MyNumpyReader(self._dtype)

    def transform(self, transform_ctx: TransformContext):
        block_size = 192    
        l = 0
        r = 256        
        start0 = np.random.randint(l,r-block_size)
        start1 = np.random.randint(l,r-block_size)    
        flip_ratio = np.random.rand(1)
        rot_angle = np.random.randint(-15, 15)
        data = np.load(transform_ctx['label'].decode('UTF-8'), allow_pickle=True)['data'].astype(self._dtype)
        block_size_z = 64
        if data.shape[2]>block_size_z:
            start_slice = np.random.randint(0,data.shape[2]-block_size_z)
        else:
            start_slice = 0
        for field in self.fields:            
            file_name = transform_ctx[field]            
            transform_ctx.set_image(field, self._reader.read(data, file_name, flip_ratio, rot_angle, start0, start1, start_slice, self._shape))
        return transform_ctx

class MyNumpyReader(object):
    """Reads Numpy files.

    Args:
        dtype: Type for data to be loaded.
    """
    def __init__(self, dtype=np.float32):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._dtype = dtype

    def read(self, data, file_name, flip_ratio, rot_angle, start0, start1, start_slice, shape: ShapeFormat):
        assert shape, "Please provide a valid shape."
        assert file_name, "Please provide a filename."

        if isinstance(file_name, (bytes, bytearray)):
            file_name = file_name.decode('UTF-8')
        #print("---------- opening np file ",file_name)
        #print(rot_angle,flip_ratio)
        data = np.load(file_name, allow_pickle=True)['data'].astype(self._dtype)
        #data = resize(data, (int(data.shape[0]/2), int(data.shape[1]/2), data.shape[2]))   
        block_size = 192
        block_size_z = 64
        if data.shape[2]>block_size_z:
            data = np.array(data)[start0:start0+block_size,start1:start1+block_size,start_slice:start_slice+block_size_z]
        else:           
            data = np.array(data)[start0:start0+block_size,start1:start1+block_size,:]
        
        if flip_ratio >= 0.5:
            data = np.fliplr(data)
        data = ndimage.rotate(data, rot_angle, reshape=False)
        
        if 'label' in file_name:
            data[data>=0.5] = 1
            data[data<0.5] = 0    
        
        data = np.expand_dims(data, -1)
        data = np.expand_dims(data, 0)
        assert len(data.shape) == shape.get_number_of_dims(), \
            "Dims of loaded data and provided shape don't match."

        img = MedicalImage(data, shape)
        img.set_property(ImageProperty.ORIGINAL_SHAPE, data.shape)
        img.set_property(ImageProperty.FILENAME, file_name)

        return img
