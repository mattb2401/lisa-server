package recognition

import (
	"path/filepath"
	"io/ioutil"
	"os"
	"bufio"
	"github.com/tensorflow/tensorflow/tensorflow/go"
	"sort"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

type Label struct {
	Label string
	Probability float32
}

type Labels []Label

func (a Labels) Len() int           { return len(a) }
func (a Labels) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a Labels) Less(i, j int) bool { return a[i].Probability > a[j].Probability }

func loadImageClassificationModel() (*tensorflow.Graph, []string, error) {
	assetsDirectory := "./assets/model_files/inception5h/"
	var (
		modelFile = filepath.Join(assetsDirectory, "tensorflow_inception_graph.pb")
		labelFile = filepath.Join(assetsDirectory, "imagenet_comp_graph_label_strings.txt")
	)
	model, err := ioutil.ReadFile(modelFile)
	if err != nil {
		return nil, nil, err
	}
	graph := tensorflow.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		return nil, nil, err
	}
	labelsFile, err := os.Open(labelFile)
	if err != nil {
		return nil, nil, err
	}
	defer labelsFile.Close()
	scanner := bufio.NewScanner(labelsFile)
	var labels []string
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	return graph, labels, nil
}

func createTensorFromImage(fileName string) (*tensorflow.Tensor, error) {
	fileBytes, err := ioutil.ReadFile(fileName)
	if err != nil {
		return nil, err
	}
	tensor, err := tensorflow.NewTensor(string(fileBytes))
	if err != nil {
		return nil, err
	}
	graph, input, output, err := getNormalizedGraph()
	if err != nil {
		return nil, err
	}
	session, err := tensorflow.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	normalized, err := session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{
			input: tensor,
		},
		[]tensorflow.Output{
			output,
		},nil)
	if err != nil {
		return nil, err
	}
	return normalized[0], nil
}

func getNormalizedGraph() (graph *tensorflow.Graph, input, output tensorflow.Output, err error) {
	s := op.NewScope()
	input = op.Placeholder(s, tensorflow.String)
	// 3 return RGB image
	decode := op.DecodeJpeg(s, input, op.DecodeJpegChannels(3))

	// Sub: returns x - y element-wise
	output = op.Sub(s,
		// make it 224x224: inception specific
		op.ResizeBilinear(s,
			// inserts a dimension of 1 into a tensor's shape.
			op.ExpandDims(s,
				// cast image to float type
				op.Cast(s, decode, tensorflow.Float),
				op.Const(s.SubScope("make_batch"), int32(0))),
			op.Const(s.SubScope("size"), []int32{224, 224})),
		// mean = 117: inception specific
		op.Const(s.SubScope("mean"), float32(117)))
	graph, err = s.Finalize()

	return graph, input, output, err
}

func ClassifyImage(imageFile string) ([]Label, error) {
	graph, labels, err := loadImageClassificationModel()
	if err != nil {
		return nil, err
	}
	tensor, err := createTensorFromImage(imageFile)
	if err != nil {
		return nil, err
	}
	session, err := tensorflow.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	output, err := session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{
			graph.Operation("input").Output(0): tensor,
		},
		[]tensorflow.Output{
			graph.Operation("output").Output(0),
		},
		nil)
	if err != nil {
		return nil, err
	}
	response := getTopFiveLabels(labels, output[0].Value().([][]float32)[0])
	return response, nil
}

func getTopFiveLabels(labels []string, probabilities []float32) []Label {
	var resultLabels []Label
	for i, p := range probabilities {
		if i >= len(labels) {
			break
		}
		resultLabels = append(resultLabels, Label{Label: labels[i], Probability: p})
	}

	sort.Sort(Labels(resultLabels))
	return resultLabels[:5]
}



