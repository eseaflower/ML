using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Reflection;
using Sectra.Client.ObjectModel;
using Sectra.Common.Imaging.Overlays;
using Sectra.DataContainers;
using Sectra.Diagnostics;
using Sectra.Imaging.Impl;
using Sectra.Imaging.Interfaces;
using Sectra.Utilities;
using Newtonsoft.Json;
using System.IO;

using PythonComm;

namespace Sectra.Client.Components {


    public class Prediction {
        public double x { get; set; }
        public double y { get; set; }        
    }

    public class PythonMammo : BasePythonContext {
        public PythonMammo()
            : base("mammo", "MammoContext") {            
        }
        public object CreateModel() {
            return this.InstanceCall<object>("createModel");
        }
        public Prediction Predict(string header, MemoryStream data) {
            return this.InstanceCall<Prediction>("predict", header, data);
        }
    }




    /// <summary>
    /// Class providing image loading and rendering for snapshot/export usage.
    /// Implements IImageDataQueueUser, to ensure that image data loading does not 
    /// interfere with loading for other purposes.
    /// </summary>
    [TraceCategory("Client.Features.Snapshot")]
    public class SnapshotImageView : ISettingsProvider, IImageDataQueueUser, IDisposable {

        #region Delegate declaration
        /// <summary>
        /// Delegate called when a bitmap export has completed.
        /// </summary>
        public delegate void BitmapCompletedCallback(Bitmap bitmap, ITransformedStreamImage sourceImage, bool cancelled, Exception e);

        #endregion

        #region Events

        /// <summary>
        /// Event fired when export is being rendered. Allows external party to render additional graphics.
        /// </summary>
        public event RenderExportEventHandler RenderSnapshot;

        /// <summary>
        /// Event that is fired when a snapshot has been rendered.
        /// Provided is a stream of the snapshot data, and the source image.
        /// </summary>
        public event RenderCompletedCallback SnapshotCompleted;

        /// <summary>
        /// Raised when an export to a bitmap is complete.
        /// </summary>
        public event BitmapCompletedCallback BitmapCompleted;
        #endregion

        #region Constants

        /// <summary>
        /// Maximum size/scale of overlays, in destination image pixels.
        /// </summary>
        private const float maxOverlayScale = 7.0f;

        /// <summary>
        /// The size of the image is compared with this threshold to determine
        /// the minimum allow scale of overlays.
        /// </summary>
        private const int minSizeThreshold = 512;

        #endregion Constants

        #region Fields

        /// <summary>
        /// The device that performs image rendering.
        /// </summary>
        private ExportDevice device = null;

        /// <summary>
        /// Handles all overlay addins.
        /// </summary>
        private readonly IOverlayRendererProvider overlayRendererProvider = null;

        /// <summary>
        /// The image manager.
        /// </summary>
        private readonly IImageDataManager imageDataManager = null;

        /// <summary>
        /// True when object has been disposed.
        /// </summary>
        private bool isDisposed = false;

        /// <summary>
        /// Provides information about visibility status for different overlays.
        /// </summary>
        private readonly OverlayVisibilityProvider overlayVisibilityProvider = null;

        /// <summary>
        /// The tick count when starting snapshot rendering, after the source image has been
        /// delivered.
        /// </summary>
        private long startTicks = 0;
        #endregion

        #region Construction and destruction

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="application">The object model root.</param>
        /// <param name="overlayVisibilityProvider">Provides information about visibility status for different overlays.</param>
        public SnapshotImageView(IApplication application, OverlayVisibilityProvider overlayVisibilityProvider) {
            this.overlayRendererProvider = application.OverlayRendererProvider;
            this.imageDataManager = application.ImageDataManager;
            this.overlayVisibilityProvider = overlayVisibilityProvider;            
            this.device = new ExportDevice(overlayVisibilityProvider, application.MemoryManager);
            this.device.RenderExport += new  ExportDevice.RenderExportEventHandlerEx(device_RenderExport);
        }

        /// <summary>
        /// Drop resources.
        /// </summary>
        public void Dispose() {            
            if (!isDisposed) {
                // Drop the device, detach from event handler.
                if (this.device != null) {
                    this.device.RenderExport -= new ExportDevice.RenderExportEventHandlerEx(device_RenderExport);
                    this.device = null;
                }

                isDisposed = true;
            }
        }

        #endregion

        #region Public properties

        /// <summary>
        /// Returns true if object has been disposed.
        /// </summary>
        public bool IsDisposed {
            get { return isDisposed; }
        }

        /// <summary>
        /// True if we allow the queue to be discarded when a new request is made.
        /// Always false for snapshots.
        /// </summary>
        public bool AllowQueueDiscard {
            get { return false; }
        }
        
        /// <summary>
        /// The order is unknown.
        /// </summary>        
        public int LoadOrder {
            get { return -1; }
        }

        #endregion

        #region ISettingsProvider methods

        /// <summary>
        /// Implementation of ISettingsProvider
        /// </summary>
        public bool ResetToSavedBaseSettings { get { return false; } }

        #endregion

        #region Public methods

        /// <summary>
        /// Render the export to a Bitmap.
        /// </summary>
        public void RenderToBitmap(ITransformedStreamImage transformedStreamImage) {
            device.RenderToBitmap(transformedStreamImage, BitmapRenderDone);                        
        }


        /// <summary>
        /// Renders a new snapshot, asynch operation.
        /// </summary>
        /// <remarks>
        /// It is possible to convert a Monochrome image to RGB, but not a RGB to Monochrome
        /// 
        /// If a monochrome->RGB conversion is performed a different rendering path is used
        /// This raises a SnapshotRGBConversionCompleted event instead of a SnapshotCompletedEvent
        /// </remarks>
        /// <param name="transformedStreamImage">The source image of the snapshot.</param>
        /// <param name="outputType"> The output type, RGB to Monochrome is not supported </param>
        /// <param name="applyPresentationWindowLevel"> Applies the currently used presentation window levling to the saved snapshot </param>
        public void RenderSnapshotAsynch(ITransformedStreamImage transformedStreamImage, ImageTypeEnum outputType, bool applyPresentationWindowLevel, string viewportMarker){            
            TraceContext tc = Trace.GetTraceContext(MethodBase.GetCurrentMethod());            
            
            if (transformedStreamImage.StreamImage.Type == ImageTypeEnum.RGB && outputType == ImageTypeEnum.MONOCHROME){
                throw new ArgumentException("a RGB image can not be saved as a monochrome snapshot");
            }
            // If we convert the image from monochrome to RGB we will reduce bit depth. 
            // Because of this we should also copy the window level settings to provide an accurate reproduction of the users image
            var imageDescriptors = GetImageDescriptors(transformedStreamImage, applyPresentationWindowLevel);

            // Check for size of caller's screen pixel, in PR space.
            var screenToPR = transformedStreamImage.PhysicalRegionToScreenTransform.Clone();
            screenToPR.Invert();
            double sourcePhysicalPixelSize = GeometryUtil.TransformDistance(1, screenToPR);
            Trace.Verbose(tc, "Screen pixel size in PR coordinates: " + sourcePhysicalPixelSize);

            // Get the data.
            imageDataManager.GetImageData(imageDescriptors, this, (data, canceled, ex) => 
                GetSnapshotDataCallback(data, canceled, ex, outputType, sourcePhysicalPixelSize, viewportMarker));

        }

        #endregion

        #region Private methods

        /// <summary>
        /// Handles RenderExport events, performs drawing of overlays.
        /// </summary>
        private void device_RenderExport(IGraphicsDevice device, ITransformedStreamImage streamImage, ExportDevice.OverlayRenderArgs overlayArguments) {

            // First fire event, to allow external parties to add content.
            if (this.RenderSnapshot != null) {
                RenderSnapshot(device, streamImage);
            }
            
            // Finally draw overlays, if any. Done last to ensure that overlays are always on top.
            RenderOverlays(device, streamImage, overlayArguments);
        }

        /// <summary>
        /// Performs rendering of overlays for the snapshot/export.
        /// </summary>
        private void RenderOverlays(IGraphicsDevice graphicsDevice, ITransformedStreamImage streamImage, ExportDevice.OverlayRenderArgs overlayArguments) {
            // Do we have any overlays to render?
            if ((streamImage == null) || !HasOverlays(streamImage, overlayArguments)) {
                return;
            }

            // Get the trace context for use later.
            TraceContext tc = Trace.GetTraceContext(MethodBase.GetCurrentMethod());
            Trace.Verbose(tc, "Snapshot image has " +streamImage.Overlays.Count + " overlays, including DICOM annotations.");

            // Create a tile descriptor and ISectraGraphics, needed by the overlay renderers.
            TileDescriptor tile = new TileDescriptor(null);
            tile.SetViewPort(0, 0, streamImage.StreamImage.Width, streamImage.StreamImage.Height);
            RectangleF viewportRectangle = new RectangleF(tile.X, tile.Y, tile.Width, tile.Height);
            tile.Image = streamImage;
            tile.SetTruePixelSize(null, null);
            ISectraGraphics graphics = new SectraGraphics(graphicsDevice);

            // Render mark before overlays, so it does not
            // obscure any graphics.
            if (overlayArguments.ViewportMarker != null) {
                // Render a mark that indicates that this is a secondary capture.
                var textMark = Text.CreateViewportMark(overlayArguments.ViewportMarker, streamImage.PhysicalRegionToScreenTransform, viewportRectangle);
                textMark.Render(PrimitiveRenderState.ReadOnly, streamImage.PhysicalRegionToScreenTransform,
                    new PrimitiveRenderArgs(graphics, OverlayRenderQuality.HighQuality, ImageViewType.Export, viewportRectangle));
            }

            (new OverlayRenderer(this.overlayRendererProvider, this.overlayVisibilityProvider)).RenderOverlays(
                streamImage, graphics, viewportRectangle, GetOverlayScaleFactor(streamImage, overlayArguments.OverlayPixelSize), 
                ImageViewType.Export);

        }


        /// <summary>
        /// Get the scale factor to use for the overlays.
        /// </summary>
        private float GetOverlayScaleFactor(ITransformedStreamImage streamImage, double overlayPixelSize) {

            TraceContext tc = Trace.GetTraceContext(System.Reflection.MethodBase.GetCurrentMethod());

            // Determine caller's screen pixel size in destination sample coordinates.
            // This is used by the device to render overlays in reasonable size.
            System.Drawing.Drawing2D.Matrix physicalToSample = streamImage.SampleToPhysicalRegionTransform.Clone();
            physicalToSample.Invert();
            float pixelSize = (float)GeometryUtil.TransformDistance(overlayPixelSize, physicalToSample);
            Trace.Verbose(tc, "Computed overlay scale: {0}", pixelSize);
            // The min size of the overlays will depend on the size of the image. This is to keep
            // CT, MR and US snapshots from having too large graphics.
            int imgSize = (int)Math.Max(streamImage.StreamImage.Width, streamImage.StreamImage.Height);
            float minOverlayScale = (imgSize > minSizeThreshold) ? 2.0f : 1.0f;
            pixelSize = (float)Math.Min(maxOverlayScale, pixelSize);
            pixelSize = (float)Math.Max(minOverlayScale, pixelSize);            
            return pixelSize;
        }

        /// <summary>
        /// Check if the image and overlay arguments for any overlays.
        /// </summary>
        private static bool HasOverlays(ITransformedStreamImage streamImage, ExportDevice.OverlayRenderArgs overlayArguments) {
            if ((streamImage.Overlays != null) && (streamImage.Overlays.Count > 0)) {
                return true;
            }
            if ((streamImage.Image.ExternalOverlayProvider != null) &&
                (streamImage.Image.ExternalOverlayProvider.GetLateRenderedBitmapOverlays().Count > 0)) {
                return true;
            }
            if (overlayArguments.ViewportMarker != null) {
                return true;
            }
            return false;
        }


        /// <summary>
        /// Renders a new snapshot, asynch operation.
        /// </summary>
        private List<IImageDataDescriptor> GetImageDescriptors(ITransformedStreamImage transformedStreamImage, bool copyWindowLevel){

            // Create a tile description that fits the view region.            
            IViewRegion viewRegion = transformedStreamImage.PrincipalViewRegion;
            IDisplayUnit displayUnit = transformedStreamImage.DisplayUnit;

            // Get a descriptor for the provided view region.
            IViewRegionDescriptor vrDescriptor = null;
            if (!displayUnit.GetViewRegionDescriptorForPosition(viewRegion.PhysicalRegion, viewRegion.Position,
                out vrDescriptor)){
                throw new Exception("Failed to get view region description");
            }
            List<IViewRegionDescriptor> vrList = new List<IViewRegionDescriptor>();
            vrList.Add(vrDescriptor);

            // Get a modifier, use this to zoom, pan etc.
            IViewRelatedRegionSettingsModifier viewModifier =
                displayUnit.GetRegionSettingsModifier(this, null, viewRegion, null);

            // Apply the settings from the image to our view modifier.
            ApplyImageSettings(viewModifier, transformedStreamImage, copyWindowLevel);
            
            // Get a descriptor suitable for export.
            //var descriptor = imageDataManager.ImageDataDescriptorFactory.GetImageDataDescriptor(displayUnit, this, vrList);
            var descriptor = imageDataManager.ImageDataDescriptorFactory.GetImageDataDescriptor(displayUnit, vrList, 50);

            if (descriptor == null){
                throw new Exception("Failed to get an image data descriptor for the snapshot");
            }

            return new List<IImageDataDescriptor>{descriptor};
        }

        /// <summary>
        /// Apply the image settings to the view modifier.
        /// </summary>
        private static void ApplyImageSettings(IViewRelatedRegionSettingsModifier viewModifier, ITransformedStreamImage transformedStreamImage, bool copyWindowLevel) {
            // Get the trace context for use later.
            TraceContext tc = Trace.GetTraceContext(MethodBase.GetCurrentMethod());

            // Reset before we start to apply any changes.
            viewModifier.ResetRegionSettings();
            // The above resets to the default setting. 
            // Ensure that crop/pan/rotate settings are completely reset
            viewModifier.ResetForExport();
            viewModifier.ZoomTrueSize(1.0);
            
            
            // Copy the window level if requested
            if (copyWindowLevel) {
                var lut = transformedStreamImage.StreamImage.ImageLut.VoiLut;
                viewModifier.SetWindowLevel(null, lut.Center, lut.Width);
            }

            if (transformedStreamImage.OrientationOperations != null) {
                foreach (ImageOrientationOperationEnum operation in transformedStreamImage.OrientationOperations) {
                    switch (operation) {
                        case ImageOrientationOperationEnum.FLIP_HORIZONTAL:
                            Trace.Verbose(tc, "Applying horizontal flip.");
                            viewModifier.FlipHorizontally();
                            break;
                        case ImageOrientationOperationEnum.FLIP_VERTICAL:
                            Trace.Verbose(tc, "Applying vertical flip.");
                            viewModifier.FlipVertically();
                            break;
                        case ImageOrientationOperationEnum.ROTATE180:
                            Trace.Verbose(tc, "Applying rotate 180.");
                            viewModifier.Rotate(RotationEnum.ABSOLUTE_CW_180);
                            break;
                        case ImageOrientationOperationEnum.ROTATE270:
                            Trace.Verbose(tc, "Applying rotate 270.");
                            viewModifier.Rotate(RotationEnum.ABSOLUTE_CW_270);
                            break;
                        case ImageOrientationOperationEnum.ROTATE90:
                            Trace.Verbose(tc, "Applying rotate 90.");
                            viewModifier.Rotate(RotationEnum.ABSOLUTE_CW_90);
                            break;
                        case ImageOrientationOperationEnum.FREE_ROTATION:
                            // We ignore the free rotation and settle for the nearest 90-degree rotation
                            break;
                    }
                }
            }
        }

        
        /// <summary>
        /// Callback called when the requested image has been received. 
        /// If all worked OK, it is now handed to the Renderer.
        /// </summary>
        private void GetSnapshotDataCallback(IImageDataCallbackData callbackData, bool cancelled, Exception exc, 
            ImageTypeEnum outputType, double sourcePhysicalPixelSize, string viewportMarker) {
            if (cancelled || (exc != null)) {
                // No need to go on. Fire event.
                if (this.SnapshotCompleted != null) {
                    this.SnapshotCompleted(null, cancelled, exc);
                }
            }
            if (!cancelled && (exc == null)) {






                var pred = WriteTrainingData(callbackData.TransformedStreamImage);





                var vr = callbackData.TransformedStreamImage.PrincipalViewRegion;
                var du = callbackData.TransformedStreamImage.DisplayUnit;
                var pending = vr.CreatePendingOverlay(BasicOverlay.Circle.TypeGuid) as IModifiableOverlay;
                bool dummy;
                var center = vr.GetCenter();
                var ext = vr.DeprecatedExtent();
                double xpos = (ext.EndX - ext.StartX) * pred.x;
                double ypos = (ext.EndY - ext.StartY) * pred.y;

                var circ = new BasicOverlay.Circle(new PointF((float)xpos, (float)ypos), 10.0);
                circ.OverlayGuid = pending.InstanceGuid;
                circ.PhysicalRegionGuid = vr.PhysicalRegion.PhysicalRegionGuid;
                circ.Extent = new ImageDataRegion.Extent4D(vr.DeprecatedExtent());
                try {
                    pending.Save(circ, du, null, out dummy);
                } catch (Exception e) {
                    Console.WriteLine(e.ToString());
                }


                
                // Start a timer, to measure the rendering/compositing time.
                startTicks = HighResClock.NowTicks;
                                                
                // This object becomes the owner of the TransformedStreamImage when we get it in this callback
                // Thus, we need to release it when we are done, or when we are disposed              
                // Create the arguments to the snapshot call.
                var arguments = new SnapshotArguments(callbackData.TransformedStreamImage, 
                    callbackData.ViewRegionDescriptor, 
                    outputType,
                    sourcePhysicalPixelSize,
                    viewportMarker);

                // Let the device render the snapshot. The
                // Render method takes ownership of the arguments.
                //device.Render(arguments, SnapshotRendered);
                SnapshotRendered(null, false, new Exception("Fake exception"));
            }
        }

        
        /// <summary>
        /// REMOVE !!!!!!!!!!!!!!!!!!
        /// </summary>
        static PythonMammo ctx = null;

        /// <summary>
        /// REMOVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        /// </summary>        
        private Prediction WriteTrainingData(ITransformedStreamImage transformedStreamImage) {

            if (ctx == null) {
                ctx = new PythonMammo();
                ctx.Construct();
                ctx.CreateModel();
            }



            if (transformedStreamImage != null) {
                Guid fileGuid = Guid.NewGuid();
                string directory = @"G:\temp\conv_net_mammo\";
                string tmpPixelFilename = String.Format("{0}{1}.pxl", directory, fileGuid);
                string tmpHeaderFilename = String.Format("{0}{1}.hdr", directory, fileGuid);

                string serializedCircle = null;
                if (transformedStreamImage.Overlays != null) {
                    foreach (var oly in transformedStreamImage.Overlays) {
                        var circle =  oly.Data as BasicOverlay.Circle;
                        if (circle != null) {
                            var xform = transformedStreamImage.SampleToPhysicalRegionTransform.Clone();
                            xform.Invert();
                            var sampleSpaceCircle = xform.TransformPoint(circle.Center);
                            int x = (int)Math.Round(sampleSpaceCircle.X);
                            int y = (int)Math.Round(sampleSpaceCircle.Y);
                            var circleData = new Dictionary<string, object>() { { "x", x }, { "y", y} };
                            serializedCircle = JsonConvert.SerializeObject(circleData);
                        }
                    }
                }


                var si = transformedStreamImage.StreamImage;
                Dictionary<string, object> header = new Dictionary<string, object>() { 
                    {"width" , si.Width}, 
                    {"height" , si.Height},
                    {"circle" , serializedCircle}
                };
                
                string serializedHeader = JsonConvert.SerializeObject(header);
                var data = new byte[transformedStreamImage.StreamImage.StrideY * si.Height];
                var ds = si.GetImageDataStream();
                ds.Read(data, 0, data.Length);
                var ms = new MemoryStream(data, true);
                return ctx.Predict(serializedHeader, ms);
                
                /*using (var f = File.OpenWrite(tmpHeaderFilename)) {
                    using (StreamWriter sw = new StreamWriter(f)) {
                        sw.Write(serializedHeader);
                    }                    
                }

                using (var f = File.OpenWrite(tmpPixelFilename)) {
                    var data = new byte[transformedStreamImage.StreamImage.StrideY * si.Height];
                    var ds = si.GetImageDataStream();
                    ds.Read(data, 0, data.Length);
                    f.Write(data, 0, data.Length);
                }*/

            }
            return null;
        }


        /// <summary>
        /// Callback that gets called when the device has finished rendering the snapshot.
        /// </summary>
        private void SnapshotRendered(ExportRenderData renderData, bool cancelled, Exception e) {

            if (this.SnapshotCompleted != null) {
                float renderTime = HighResClock.TicksToMs(HighResClock.NowTicks - startTicks);
                Trace.Verbose(MethodBase.GetCurrentMethod(), "Snapshot rendering took {0} ms, excluding image transfer time.",
                    renderTime);

                if (renderData == null && e == null) {
                    e = new Exception("Failed to render image");
                }

                this.SnapshotCompleted(renderData, cancelled, e);
            } else {
                // Dispose the renderData if it is not beeing used.
                if (renderData != null) {
                    renderData.Dispose();
                }
            }
        }
        
        /// <summary>
        /// Called when the bitmap is ready.
        /// </summary>
        private void BitmapRenderDone(BitmapRenderData bitmapData, bool cancelled, Exception e) {
            if (this.BitmapCompleted != null) {
                if ((bitmapData == null) || cancelled || (e != null)) {
                    this.BitmapCompleted(null, null, cancelled, e);
                } else {
                    this.BitmapCompleted(bitmapData.Bitmap, bitmapData.SourceImage, false, null);
                }
            }
        }

        #endregion

    }

}
