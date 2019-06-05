package main

import (
	"lisa-server/recognition"
	"io/ioutil"
	"log"
)

func main() {
	imageFile, err := ioutil.ReadFile("assets/images/cat.jpeg")
	if err != nil {
		panic(err)
	}
	response, err := recognition.ClassifyImage(string(imageFile))
	if err != nil {
		panic(err)
	}
	log.Printf("response: %v", response)
}