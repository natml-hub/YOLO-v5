/* 
*   YOLO v5
*   Copyright © 2023 NatML Inc. All Rights Reserved.
*/

namespace NatML.Examples {

    using UnityEngine;
    using NatML;
    using NatML.Devices;
    using NatML.Devices.Outputs;
    using NatML.Features;
    using NatML.Vision;
    using NatML.Visualizers;

    public sealed class YOLOv5Sample : MonoBehaviour {

        [Header(@"UI")]
        public YOLOv5Visualizer visualizer;

        private CameraDevice cameraDevice;
        private TextureOutput cameraTextureOutput;

        private MLModelData modelData;
        private MLModel model;
        private YOLOv5Predictor predictor;

        async void Start () {
            // Request camera permissions
            var permissionStatus = await MediaDeviceQuery.RequestPermissions<CameraDevice>();
            if (permissionStatus != PermissionStatus.Authorized) {
                Debug.LogError(@"User did not grant camera permissions");
                return;
            }
            // Get the default camera device
            var query = new MediaDeviceQuery(MediaDeviceCriteria.CameraDevice);
            cameraDevice = query.current as CameraDevice;
            // Start the camera preview
            cameraDevice.previewResolution = (1280, 720);
            cameraTextureOutput = new TextureOutput();
            cameraDevice.StartRunning(cameraTextureOutput);
            visualizer.image = await cameraTextureOutput;
            // Create the YOLOv5 predictor
            Debug.Log("Fetching model data from NatML...");
            modelData = await MLModelData.FromHub("@natml/yolo-v5");
            model = modelData.Deserialize();
            predictor = new YOLOv5Predictor(model, modelData.labels);
        }

        void Update () {
            // Check that the predictor has been created
            if (predictor == null)
                return;
            // Predict
            var imageFeature = new MLImageFeature(cameraTextureOutput.texture);
            imageFeature.aspectMode = modelData.aspectMode;
            var detections = predictor.Predict(imageFeature);
            // Visualize
            visualizer.Render(detections);
        }

        void OnDisable () {
            // Dispose model
            model?.Dispose();
        }
    }
}