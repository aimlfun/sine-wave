using System.Diagnostics;
using System.Drawing.Imaging;

namespace SinWave
{
    public partial class Form1 : Form
    {
        /// <summary>
        /// Our NN.
        /// </summary>
        private readonly NeuralNetwork neuralNetwork;

        /// <summary>
        /// Where it will read / write the AI model file.
        /// </summary>
        private const string c_aiModelFilePath = @"c:\temp\sinWave.ai";

        /// <summary>
        /// Number of sides in the polygon.
        /// </summary>
        private int numberOfSidesPolygonHas = 3;

        /// <summary>
        /// We adjust this so each polygon is rotated 10 degrees
        /// </summary>
        private int rotationalOffsetForPolygon = 0;

        /// <summary>
        /// Constructor.
        /// </summary>
        public Form1()
        {
            InitializeComponent();
            Show();

            int[] layers = new int[6] { 1 /* INPUT: angle/360 */,
                                        90, 90, 90, 90,
                                        1 /* OUTPUT: Math.Sin(angle) */ 
                                       };

            // TanH is my preferred function, so far I've seen limited potential in the others.
            ActivationFunctions[] activationFunctions = new ActivationFunctions[6] { ActivationFunctions.TanH, ActivationFunctions.TanH,
                                                                                     ActivationFunctions.TanH, ActivationFunctions.TanH,
                                                                                     ActivationFunctions.TanH, ActivationFunctions.TanH};

            neuralNetwork = new(0, layers, activationFunctions, false);
            Train();

            /*          
                        // uncomment this to plot the sin wave as a graph.
                        Bitmap b = new(720, 200);

                        for (int x = 0; x < 720; x++)
                        {
                            float y = SinViaAI(x/2)*100+100;
                            b.SetPixel(x, (int) Math.Round(y ), Color.Black);
                        }

                        b.Save(@"c:\temp\sine-wave.png", ImageFormat.Png);
            */

            timer1.Enabled = true;
            timer1.Tick += Timer1_Tick;
            timer1.Start();
        }

        /// <summary>
        /// Draws a polygon of increasing sides, using AI/ML Math.Sin()
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Timer1_Tick(object? sender, EventArgs e)
        {
            DrawPolygon(numberOfSidesPolygonHas);

            // increase the number of sides, resetting to 3.
            if (numberOfSidesPolygonHas++ > 15) numberOfSidesPolygonHas = 3;

            // make each one drawn at 10 degrees more than the last polygon.
            rotationalOffsetForPolygon += 10;
        }

        /// <summary>
        /// Logic requires radians but we track angles in degrees, this converts.
        /// </summary>
        /// <param name="angleInDegrees">Angle</param>
        /// <returns>Angle in radians.</returns>
        internal static double DegreesInRadians(double angleInDegrees)
        {
            return Math.PI * angleInDegrees / 180;
        }

        /// <summary>
        /// Draw a polygon to prove that our AI returns SIN.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void DrawPolygon(int sides)
        {
            Bitmap polygonBitmap = pictureBox1.Image is null ? new(pictureBox1.Width, pictureBox1.Height) : new(pictureBox1.Image);

            using Graphics graphics = Graphics.FromImage(polygonBitmap);

            graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
            graphics.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;

            List<PointF> p = new();
            Random random = new();
            for (float angle = 0; angle < 360; angle += 360 / (float)sides)
            {
                float r = 200 - sides * 10;
                p.Add(new PointF(pictureBox1.Width / 2 + r * SinViaAI((angle + rotationalOffsetForPolygon) % 360),
                                 pictureBox1.Height / 2 + r * CosViaAI((angle + rotationalOffsetForPolygon) % 360)));
            }

            // use a random colour brush
            using SolidBrush brush = new(Color.FromArgb(40, random.Next(255), random.Next(255), random.Next(255)));

            graphics.FillPolygon(brush, p.ToArray());
            p.Add(p.ToArray()[0]);
            graphics.DrawLines(Pens.White, p.ToArray());

            pictureBox1.Image?.Dispose();
            pictureBox1.Image = polygonBitmap;
        }

        /// <summary>
        /// Returns Math.Sin(), using AI!
        /// </summary>
        /// <param name="angle">Angle (0..1)</param>
        /// <returns></returns>
        private float SinViaAI(float angle)
        {
            if (angle < 0 || angle >= 360) throw new ArgumentOutOfRangeException(nameof(angle), "0<=x<360");

            return neuralNetwork.FeedForward(new float[] { angle / 360F })[0];
        }

        /// <summary>
        /// Returns Math.Cos(), using AI. 
        /// Cos(a) == Sin(a+90).
        /// </summary>
        /// <param name="angle">Angle (0..1)</param>
        /// <returns></returns>
        private float CosViaAI(float angle)
        {
            return SinViaAI((angle + 90F) % 360F);
        }

        /// <summary>
        /// Train the AI to return Math.Sin().
        /// </summary>
        void Train()
        {
            // load a pre-trained model if found.
            if (File.Exists(c_aiModelFilePath))
            {
                neuralNetwork.Load(c_aiModelFilePath);
                Text = "Sin Wave - MODEL LOADED"; // we know it passed training
                return;
            }

            bool trained = false;

            float[] angleDiv360toBetween0and1 = new float[360];
            float[] sinOutputForAngle = new float[360];

            // do this once, and cache as Sin() is slow.
            for (int n = 0; n < 360; n++)
            {
                angleDiv360toBetween0and1[n] = n / 360F;
                sinOutputForAngle[n] = (float)Math.Sin(DegreesInRadians(n));
            }

            // train the AI up to 50k times, exiting if we are getting correct answers.
            // note: it could fail even with 50k, simply because it picks bad initial weights and biases.
            for (int epoch = 0; epoch < 50000; epoch++)
            {
                for (int n = 0; n < 360; n++)
                {
                    float[] inputs = new float[] { angleDiv360toBetween0and1[n] };

                    neuralNetwork.BackPropagate(inputs, new float[] { sinOutputForAngle[n] });
                }

                // by this point we *may* have done enough training...
                // we check the output, and if it's accurate, we exit. We don't check before 5k,
                // because it would slow training down for little gain.
                if (epoch > 5000)
                {
                    trained = true;

                    // test the result for all permutations, and if all are good, we're trained.
                    for (int n = 0; n < 360; n++)
                    {
                        float[] inputs = new float[] { angleDiv360toBetween0and1[n] };

                        float expectedResult = sinOutputForAngle[n];
                        float prediction = neuralNetwork.FeedForward(inputs)[0];

                        if (Math.Abs(expectedResult - prediction) > 0.01F) // close enough (accuracy)
                        {
                            // enable this to see how close it gets...
                            // Debug.WriteLine($"{n} AI:{prediction:0.000} Expected: {expectedResult:0.000}");
                            trained = false; // wrong answer
                            break;
                        }
                    }

                    if (trained)
                    {
                        Text = "Sin Wave - TRAINED."; // we know it passed training
                        neuralNetwork.Save(c_aiModelFilePath);
                        break;
                    }
                }

                if (epoch % 1000 == 0) // indicator of progress, every 1000
                {
                    Text = $"Sin Wave - TRAINING. GENERATION {epoch}";
                    Application.DoEvents();
                }
            }

            // Remember back propagation is about finding the right weights/biases to provide the desired outcome. We pick the initial value
            // at random, and sometimes we start from a bad initial weights that takes much more iterations. Rather than try forever, we give
            // up at 50k, and ask the user to re-run.
            if (!trained) MessageBox.Show("Unable to train successfully (poor initial random weights/biases), please re-run.");
        }
    }
}