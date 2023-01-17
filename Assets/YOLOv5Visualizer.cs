/* 
*   YOLO v5
*   Copyright © 2023 NatML Inc. All Rights Reserved.
*/

namespace NatML.Visualizers {

    using System.Collections.Generic;
    using UnityEngine;
    using UnityEngine.UI;

    /// <summary>
    /// </summary>
    [RequireComponent(typeof(RawImage), typeof(AspectRatioFitter))]
    public sealed class YOLOv5Visualizer : MonoBehaviour {

        #region --Inspector--
        /// <summary>
        /// Detection rectangle prefab.
        /// </summary>
        [Tooltip(@"Detection rectangle prefab.")]
        public YOLOv5Detection detection;
        #endregion


        #region --Client API--
        /// <summary>
        /// Detection source image.
        /// </summary>
        public Texture2D image {
            get => rawImage.texture as Texture2D;
            set {
                rawImage.texture = value;
                aspectFitter.aspectRatio = (float)value.width / value.height;
            }
        }

        /// <summary>
        /// Render a set of object detections.
        /// </summary>
        /// <param name="image">Image which detections are made on.</param>
        /// <param name="detections">Detections to render.</param>
        public void Render (params (Rect rect, string label, float score)[] detections) { // INCOMPLETE
            // Delete current
            foreach (var rect in currentRects)
                GameObject.Destroy(rect.gameObject);
            currentRects.Clear();
            // Render rects
            foreach (var d in detections) {
                var rect = Instantiate(detection, transform);
                rect.gameObject.SetActive(true);
                rect.Render(rawImage, d.rect, d.label, d.score);
                currentRects.Add(rect);
            }
        }
        #endregion


        #region --Operations--
        RawImage rawImage;
        AspectRatioFitter aspectFitter;
        readonly List<YOLOv5Detection> currentRects = new List<YOLOv5Detection>();

        void Awake () {
            rawImage = GetComponent<RawImage>();
            aspectFitter = GetComponent<AspectRatioFitter>();
        }
        #endregion
    }
}