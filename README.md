![Tests](https://github.com/j178/tiktoken-go/actions/workflows/go.yml/badge.svg)

# tiktoken-go

> [!NOTE]
> This is a fork of [tiktoken-go/tokenizer](https://github.com/tiktoken-go/tokenizer) with some API changes.

This is a pure go port of OpenAI's tokenizer.

<a href="https://www.buymeacoffee.com/mwahlmann" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-blue.png" alt="Buy Me A Coffee" height="41" width="174"></a>

## Usage

```go
package main

import (
	"fmt"

	"github.com/j178/tiktoken-go"
)

func main() {
	enc, err := tiktoken.Get(tiktoken.Cl100kBase)
	if err != nil {
		panic("oh oh")
	}

	// this should print a list of token ids
	ids, _, _ := enc.Encode("supercalifragilistic")
	fmt.Println(ids)

	// this should print the original string back
	text, _ := enc.Decode(ids)
	fmt.Println(text)
}
```

Alternatively you can use the included command-line tool

```sh
> tokenizer -h

Usage of tokenizer:
  -decode string
        tokens to decode
  -encode string
        text to encode
  -token string
        text to calculate token

> tokenizer -encode supercalifragilistic
```

## Todo

- ✅ port code
- ✅ cl100k_base encoding
- ✅ r50k_base encoding
- ✅ p50k_base encoding
- ✅ p50k_edit encoding
- ✅ tests
- ❌ handle special tokens
- ❌ gpt-2 model

## Caveats

This library embeds OpenAI's vocabularies—which are not small (~4Mb)— as go
maps. This is different than what the way python version of tiktoken works, 
which downloads the dictionaries and puts them in a cache folder.

However, since the dictionaries are compiled during the go build process
the performance and start-up times should be better than downloading and loading
them at runtime.

## Alternatives

Here is a list of other libraries that do something similar.

- [https://github.com/sugarme/tokenizer](https://github.com/sugarme/tokenizer) (A different tokenizer algorithm than OpenAI's)
- [https://github.com/pandodao/tokenizer-go](https://github.com/pandodao/tokenizer-go) (deprecated, calls into JavaScript)
- [https://github.com/pkoukk/tiktoken-go](https://github.com/pkoukk/tiktoken-go)
