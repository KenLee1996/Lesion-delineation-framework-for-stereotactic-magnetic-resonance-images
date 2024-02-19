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
        flip_ratio = np.random.rand(1)
        rot_angle = np.random.randint(-50, 50)
        for field in self.fields:
            file_name = transform_ctx[field]
            transform_ctx.set_image(field, self._reader.read(file_name, flip_ratio, rot_angle, self._shape))
        return transform_ctx

class MyNumpyReader(object):
    """Reads Numpy files.

    Args:
        dtype: Type for data to be loaded.
    """
    def __init__(self, dtype=np.float32):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._dtype = dtype

    def read(self, file_name, flip_ratio, rot_angle, shape: ShapeFormat):
        assert shape, "Please provide a valid shape."
        assert file_name, "Please provide a filename."

        if isinstance(file_name, (bytes, bytearray)):
            file_name = file_name.decode('UTF-8')
        #print("---------- opening np file ",file_name)
        data = np.load(file_name, allow_pickle=True)['data'].astype(self._dtype)     
        
        if flip_ratio >= 0.5:
            data = np.fliplr(data)
        data = ndimage.rotate(data, rot_angle, reshape=False)
        
        if 'image' in file_name:            
            data_flat = data.flatten()
            per_99 = np.percentile(data_flat,99)
            data[np.where(data>per_99)] = per_99
        elif 'label' in file_name:
            data[data>=0.5] = 1
            data[data<0.5] = 0 
        
        data = np.expand_dims(data, 0)
        data = np.expand_dims(data, 0)

        assert len(data.shape) == shape.get_number_of_dims(), \
            "Dims of loaded data and provided shape don't match."

        img = MedicalImage(data, shape)
        img.set_property(ImageProperty.ORIGINAL_SHAPE, data.shape)
        img.set_property(ImageProperty.FILENAME, file_name)

        return img
