import numpy as np
import sys
import os
import json
# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]
    
    return np.array(keep)

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT_RES")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
        
    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output0_dtype = self.output0_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT_CONFS
            in_confs = pb_utils.get_input_tensor_by_name(request, "INPUT_CONFS")
            # Get INPUT_BOXES
            in_boxes = pb_utils.get_input_tensor_by_name(request, "INPUT_BOXES") 
            

            confs_output_array = in_confs.as_numpy()
            boxes_output_array = in_boxes.as_numpy()

            print("confs:")
            print(confs_output_array.shape)
            print("boxes:")
            print(boxes_output_array.shape)
            '''
            if len(confs_output_array) != batch_size or len(boxes_output_array) != batch_size:
                raise Exception("expected {} results, got confs {} boxes {}".format(
                    batch_size, len(confs_output_array), len(boxes_output_array)))
            '''
           
            # Include special handling for non-batching models

            # [batch, num, 1, 4]
            box_array = boxes_output_array
            # [batch, num, num_classes]
            confs = confs_output_array
            num_classes = confs.shape[2]

            box_array = box_array[:, :, 0]


            # [batch, num, num_classes] --> [batch, num]
            max_conf = np.max(confs, axis=2)
            max_id = np.argmax(confs, axis=2)

            conf_thresh=0.8
            nms_thresh=0.5
            bboxes_batch = []
            for i in range(box_array.shape[0]):
            
                argwhere = max_conf[i] > conf_thresh
                l_box_array = box_array[i, argwhere, :]
                l_max_conf = max_conf[i, argwhere]
                l_max_id = max_id[i, argwhere]

                bboxes = []
                # nms for each class
                for j in range(num_classes):

                    cls_argwhere = l_max_id == j
                    ll_box_array = l_box_array[cls_argwhere, :]
                    ll_max_conf = l_max_conf[cls_argwhere]
                    ll_max_id = l_max_id[cls_argwhere]

                    keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)
                    
                    if (keep.size > 0):
                        ll_box_array = ll_box_array[keep, :]
                        ll_max_conf = ll_max_conf[keep]
                        ll_max_id = ll_max_id[keep]

                        for k in range(ll_box_array.shape[0]):
                            bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])
                bboxes = np.array(bboxes)
                bboxes = bboxes.reshape(-1)
                
                bboxes_batch.append([bboxes.tobytes()])
                
            print("bboxes_batch")
            print(bboxes_batch)
            bboxes_batch = np.array(bboxes_batch)
            print(bboxes_batch.shape)
            out_tensor_0 = pb_utils.Tensor("OUTPUT_RES",
                                           bboxes_batch)

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')