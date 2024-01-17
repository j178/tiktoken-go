package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/j178/tiktoken-go"
)

func main() {
	model := flag.String("model", "gpt-3.5-turbo", "the target OpenAI model to generate tokens for")
	encoding := flag.String("encoding", "", "the encoding format. (not that you can't specify both model and encoding)")
	encode := flag.String("encode", "", "text to encode")
	decode := flag.String("decode", "", "space separated list of token ids to decode")
	emitTokens := flag.Bool("tokens", false, "if true will output the tokens instead of the token ids")
	listEncodings := flag.Bool("list-encodings", false, "list all supported encoding formats")
	flag.Parse()

	if *listEncodings {
		printEncodings()
		os.Exit(0)
	}

	// either model or encoding should be specified
	if (*model != "" && *encoding != "") || (*model == "" && *encoding == "") {
		flag.PrintDefaults()
	}

	// either encode or decode operations should be requested
	if (*encode != "" && *decode != "") || (*encode == "" && *decode == "") {
		flag.PrintDefaults()
	}

	codec := getCodec(*model, *encoding)

	if *encode != "" {
		encodeInput(codec, *encode, *emitTokens)
	} else {
		decodeInput(codec, *decode+" "+strings.Join(flag.Args(), " "))
	}
}

func getCodec(model, encoding string) tiktoken.Codec {
	if model != "" {
		c, err := tiktoken.ForModel(model)
		if err != nil {
			log.Fatalf("error creating tokenizer: %v", err)
		}
		return c
	} else {
		c, err := tiktoken.Get(tiktoken.Encoding(encoding))
		if err != nil {
			log.Fatalf("error creating tokenizer: %v", err)
		}
		return c
	}
}

func encodeInput(codec tiktoken.Codec, text string, wantTokens bool) {
	ids, tokens, err := codec.Encode(text)
	if err != nil {
		log.Fatalf("error encoding: %v", err)
	}

	if wantTokens {
		fmt.Println(strings.Join(tokens, " "))
	} else {
		var textIds []string
		for _, id := range ids {
			textIds = append(textIds, strconv.Itoa(int(id)))
		}
		fmt.Println(strings.Join(textIds, " "))
	}
}

func decodeInput(codec tiktoken.Codec, tokens string) {
	var ids []uint
	for _, t := range strings.Split(tokens, " ") {
		id, err := strconv.Atoi(t)
		if err != nil {
			log.Fatalf("invalid token id: %s", t)
		}
		ids = append(ids, uint(id))
	}

	text, err := codec.Decode(ids)
	if err != nil {
		log.Fatalf("error decoding: %v", err)
	}
	fmt.Println(text)
}

func printEncodings() {
	encodings := []tiktoken.Encoding{
		tiktoken.R50kBase,
		tiktoken.P50kBase,
		tiktoken.P50kEdit,
		tiktoken.Cl100kBase,
	}

	for _, e := range encodings {
		fmt.Println(e)
	}
}
