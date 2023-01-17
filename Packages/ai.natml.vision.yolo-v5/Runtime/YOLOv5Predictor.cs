/* 
*   YOLO v5
*   Copyright © 2023 NatML Inc. All Rights Reserved.
*/

namespace NatML.Vision {

    using System;
    using System.Collections.Generic;
    using System.Linq;
    using UnityEngine;
    using NatML.Features;
    using NatML.Internal;
    using NatML.Types;

    /// <summary>
    /// YOLO v5 predictor for general object detection.
    /// This predictor accepts an image feature and produces a list of detections.
    /// Each detection is comprised of a normalized rect, label, and detection score.
    /// </summary>
    public sealed class YOLOv5Predictor : IMLPredictor<(Rect rect, string label, float score)[]> {

        #region --Client API--
        /// <summary>
        /// Class labels.
        /// </summary>
        public readonly string[] labels;

        /// <summary>
        /// Create the YOLO v5 predictor.
        /// </summary>
        /// <param name="model">YOLO v5 ML model.</param>
        /// <param name="labels">Classification labels.</param>
        /// <param name="minScore">Minimum candidate score.</param>
        /// <param name="maxIoU">Maximum intersection-over-union score for overlap removal.</param>
        public YOLOv5Predictor (MLModel model, string[] labels, float minScore = 0.4f, float maxIoU = 0.5f) {
            this.model = model as MLEdgeModel;
            this.labels = labels;
            this.minScore = minScore;
            this.maxIoU = maxIoU;
            this.inputType = model.inputs[0] as MLImageType;
        }

        /// <summary>
        /// Detect objects in an image.
        /// </summary>
        /// <param name="inputs">Input image.</param>
        /// <returns>Detected objects.</returns>
        public unsafe (Rect rect, string label, float score)[] Predict (params MLFeature[] inputs) {
            // Check
            if (inputs.Length != 1)
                throw new ArgumentException(@"YOLO v5 predictor expects a single feature", nameof(inputs));
            // Check type
            var input = inputs[0];
            var imageType = MLImageType.FromType(input.type);
            var imageFeature = input as MLImageFeature;
            if (!imageType)
                throw new ArgumentException(@"YOLO v5 predictor expects an an array or image feature", nameof(inputs));
            // Predict
            using var inputFeature = (input as IMLEdgeFeature).Create(inputType);
            using var outputFeatures = model.Predict(inputFeature);
            // Marshal
            var logitsData = (float*)outputFeatures[0].data;      // (1,6300,85)
            var shape8 = new [] { inputType.height / 8, inputType.width / 8, 3, 85 };
            var shape16 = new [] { inputType.height / 16, inputType.width / 16, 3, 85 };
            var shape32 = new [] { inputType.height / 32, inputType.width / 32, 3, 85 };
            var logits8 = new MLArrayFeature<float>(&logitsData[0], shape8);
            var logits16 = new MLArrayFeature<float>(&logitsData[logits8.elementCount], shape16);
            var logits32 = new MLArrayFeature<float>(&logitsData[logits8.elementCount + logits16.elementCount], shape32);
            var candidateBoxes = new List<Rect>();
            var candidateScores = new List<float>();
            var candidateLabels = new List<string>();
            foreach (var logits in new [] { logits8, logits16, logits32 })
                for (int j = 0, jlen = logits.shape[0], ilen = logits.shape[1], clen = logits.shape[2]; j < jlen; ++j)
                    for (var i = 0; i < ilen; ++i)
                        for (var c = 0; c < clen; ++c) {
                            // Check
                            var score = logits[j,i,c,4];
                            if (score < minScore)
                                continue;
                            // Get class
                            var label = Enumerable
                                .Range(5, 80)
                                .Aggregate((p, q) => logits[j,i,c,p] > logits[j,i,c,q] ? p : q) - 5;
                            // Decode box
                            var cx = logits[j,i,c,0];
                            var cy = 1f - logits[j,i,c,1];
                            var w = logits[j,i,c,2];
                            var h = logits[j,i,c,3];
                            var rawBox = new Rect(cx - 0.5f * w, cy - 0.5f * h, w, h);
                            var box = imageFeature?.TransformRect(rawBox, inputType) ?? rawBox;
                            // Add
                            candidateBoxes.Add(box);
                            candidateScores.Add(score);
                            candidateLabels.Add(labels[label]);
                        }
            var keepIdx = MLImageFeature.NonMaxSuppression(candidateBoxes, candidateScores, maxIoU);
            var result = new List<(Rect, string, float)>();
            foreach (var idx in keepIdx)
                result.Add((candidateBoxes[idx], candidateLabels[idx], candidateScores[idx]));
            // Return
            return result.ToArray();
        }
        #endregion


        #region --Operations--
        private readonly MLEdgeModel model;
        private readonly float minScore;
        private readonly float maxIoU;
        private readonly MLImageType inputType;

        void IDisposable.Dispose () { } // Not used
        #endregion
    }
}