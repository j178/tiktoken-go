package tokenizer

// Package tokenizer provides functions for encoding and decoding text using
// different tokenization schemes.
//
// Encoding Formats
//
// The following encoding formats are supported:
// - Cl100kBase
// - R50kBase
// - P50kBase
// - P50kEdit
//
// Alternatively you can request a tokenizer using OpenAI's model name, the
// following OpenAI models are supported:
// - GPT4
// - GPT35Turbo
// - TextEmbeddingAda002
// - TextDavinci003
// - TextDavinci002
// - CodeDavinci002
// - CodeDavinci001
// - CodeCushman002
// - CodeCushman001
// - DavinciCodex
// - CushmanCodex
// - TextDavinci001
// - TextCurie001
// - TextBabbage001
// - TextAda001
// - Davinci
// - Curie
// - Babbage
// - Ada
// - TextSimilarityDavinci001
// - TextSimilarityCurie001
// - TextSimilarityBabbage001
// - TextSimilarityAda001
// - TextSearchDavinciDoc001
// - TextSearchCurieDoc001
// - TextSearchAdaDoc001
// - TextSearchBabbageDoc001
// - CodeSearchBabbageCode001
// - CodeSearchAdaCode001
// - TextDavinciEdit001
// - CodeDavinciEdit001
//
// Usage Example
//
// Here is an example of how to encode a string using the `ForModel` function:
//
//	package main
//
//	import (
//		"fmt"
//		"github.com/tiktoken-go/tokenizer"
//	)
//
//	func main() {
//		enc, err := tokenizer.Get(tokenizer.Cl100kBase)
//		if err != nil {
//			panic("oh oh")
//		}
//
//		// this should print a list of token ids
//		ids, token, _ := enc.Encode("supercalifragilistic")
//		fmt.Println(ids)
//
//		// this should print the original string back
//		text, _ := enc.Decode(ids)
//		fmt.Println(text)
// }

import (
	"errors"
	"strings"

	"github.com/tiktoken-go/tokenizer/codec"
)

var (
	ErrModelNotSupported    = errors.New("model not supported")
	ErrEncodingNotSupported = errors.New("encoding not supported")
)

type Codec interface {
	GetName() string
	Encode(string) ([]uint, []string, error)
	Decode([]uint) (string, error)
}

type Encoding string

const (
	GPT2Enc    Encoding = "gpt2"
	R50kBase   Encoding = "r50k_base"
	P50kBase   Encoding = "p50k_base"
	P50kEdit   Encoding = "p50k_edit"
	Cl100kBase Encoding = "cl100k_base"
)

// Get returns a new instance of a Codec implementation based on the specified
// encoding format. The returned Codec instance can be used to encode (tokenize)
// and decode (reassemble) text. If the specified encoding is not supported,
// an error is returned.
func Get(encoding Encoding) (Codec, error) {
	switch encoding {
	case Cl100kBase:
		return codec.NewCl100kBase(), nil
	case R50kBase:
		return codec.NewR50kBase(), nil
	case P50kBase:
		return codec.NewP50kBase(), nil
	case P50kEdit:
		return codec.NewP50kEdit(), nil
	default:
		return nil, ErrEncodingNotSupported
	}
}

// Reference: https://github.com/openai/tiktoken/blob/main/tiktoken/model.py

var modelPrefixToEncoding = map[string]Encoding{
	// chat
	"gpt-4-":         "cl100k_base", // e.g., gpt-4-0314, etc., plus gpt-4-32k
	"gpt-3.5-turbo-": "cl100k_base", // e.g, gpt-3.5-turbo-0301, -0401, etc.
	"gpt-35-turbo-":  "cl100k_base", // Azure deployment name
	// fine-tuned
	"ft:gpt-4":         "cl100k_base",
	"ft:gpt-3.5-turbo": "cl100k_base",
	"ft:davinci-002":   "cl100k_base",
	"ft:babbage-002":   "cl100k_base",
}

var modelToEncoding = map[string]Encoding{
	// chat
	"gpt-4":         "cl100k_base",
	"gpt-3.5-turbo": "cl100k_base",
	"gpt-35-turbo":  "cl100k_base", // Azure deployment name
	// base
	"davinci-002": "cl100k_base",
	"babbage-002": "cl100k_base",
	// embeddings
	"text-embedding-ada-002": "cl100k_base",
	// DEPRECATED MODELS
	// text (DEPRECATED)
	"text-davinci-003": "p50k_base",
	"text-davinci-002": "p50k_base",
	"text-davinci-001": "r50k_base",
	"text-curie-001":   "r50k_base",
	"text-babbage-001": "r50k_base",
	"text-ada-001":     "r50k_base",
	"davinci":          "r50k_base",
	"curie":            "r50k_base",
	"babbage":          "r50k_base",
	"ada":              "r50k_base",
	// code (DEPRECATED)
	"code-davinci-002": "p50k_base",
	"code-davinci-001": "p50k_base",
	"code-cushman-002": "p50k_base",
	"code-cushman-001": "p50k_base",
	"davinci-codex":    "p50k_base",
	"cushman-codex":    "p50k_base",
	// edit (DEPRECATED)
	"text-davinci-edit-001": "p50k_edit",
	"code-davinci-edit-001": "p50k_edit",
	// old embeddings (DEPRECATED)
	"text-similarity-davinci-001":  "r50k_base",
	"text-similarity-curie-001":    "r50k_base",
	"text-similarity-babbage-001":  "r50k_base",
	"text-similarity-ada-001":      "r50k_base",
	"text-search-davinci-doc-001":  "r50k_base",
	"text-search-curie-doc-001":    "r50k_base",
	"text-search-babbage-doc-001":  "r50k_base",
	"text-search-ada-doc-001":      "r50k_base",
	"code-search-babbage-code-001": "r50k_base",
	"code-search-ada-code-001":     "r50k_base",
	// open source
	"gpt2": "gpt2",
}

// EncodingForModel returns the encoding format for the specified OpenAI model.
// If the specified model is not supported, an error is returned.
func EncodingForModel(model string) (Encoding, error) {
	if encoding, ok := modelToEncoding[model]; ok {
		return encoding, nil
	}
	for prefix, encoding := range modelPrefixToEncoding {
		if strings.HasPrefix(model, prefix) {
			return encoding, nil
		}
	}
	return "", ErrModelNotSupported
}

// ForModel returns a new instance of a Codec implementation based on the
// specified OpenAI model. If the specified model is not supported, an error
// is returned.
func ForModel(model string) (Codec, error) {
	enc, err := EncodingForModel(model)
	if err != nil {
		return nil, err
	}
	return Get(enc)
}
