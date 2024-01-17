package codec

import (
	"fmt"
	"math"
	"sync"

	"github.com/dlclark/regexp2"
)

type Codec struct {
	vocabulary        vocab
	reverseVocabulary reverse
	specialTokens     map[string]uint
	splitRegexp       *regexp2.Regexp
	name              string
}

func (c *Codec) GetName() string {
	return c.name
}

func (c *Codec) Encode(input string) ([]uint, []string, error) {
	var (
		ids    []uint
		tokens []string
	)
	match, err := c.splitRegexp.FindStringMatch(input)
	if err != nil {
		return nil, nil, fmt.Errorf("error matching: %v", err)
	}

	for match != nil {
		piece := match.String()
		if id, ok := c.vocabulary[piece]; ok {
			ids = append(ids, id)
			tokens = append(tokens, piece)
		} else {
			newIds, newTokens := c.bpe([]byte(piece))
			ids = append(ids, newIds...)
			tokens = append(tokens, newTokens...)
		}
		m, err := c.splitRegexp.FindNextMatch(match)
		if err != nil {
			return nil, nil, fmt.Errorf("error matching: %v", err)
		}
		match = m
	}
	return ids, tokens, nil
}

func (c *Codec) Count(input string) (int, error) {
	var count int
	match, err := c.splitRegexp.FindStringMatch(input)
	if err != nil {
		return 0, fmt.Errorf("error matching: %v", err)
	}

	for match != nil {
		piece := match.String()
		if _, ok := c.vocabulary[piece]; ok {
			count++
		} else {
			_, newTokens := c.bpe([]byte(piece))
			count += len(newTokens)
		}
		m, err := c.splitRegexp.FindNextMatch(match)
		if err != nil {
			return 0, fmt.Errorf("error matching: %v", err)
		}
		match = m
	}
	return count, nil
}

func (c *Codec) Decode(tokens []uint) (string, error) {
	if c.reverseVocabulary == nil {
		c.reverseVocabulary = make(map[uint]string)
		for k, v := range c.vocabulary {
			c.reverseVocabulary[v] = k
		}
	}

	var out string
	for _, t := range tokens {
		piece, ok := c.reverseVocabulary[t]
		if !ok {
			return "", fmt.Errorf("invalid token: %d", t)
		}
		out += piece
	}
	return out, nil
}

type part struct {
	offset int
	rank   uint
}

var partPool = sync.Pool{
	New: func() any {
		return make([]part, 0, 128)
	},
}

func (c *Codec) bpe(piece []byte) ([]uint, []string) {
	parts := partPool.Get().([]part)
	if cap(parts) < len(piece)+1 {
		parts = make([]part, len(piece)+1)
	}
	parts = parts[:len(piece)+1]
	defer func() {
		if cap(parts) <= 1024 {
			partPool.Put(parts)
		}
	}()

	for i := 0; i < len(parts); i++ {
		parts[i] = part{i, math.MaxUint}
	}

	getRank := func(index, skip int) uint {
		if index+skip+2 < len(parts) {
			start := parts[index].offset
			end := parts[index+skip+2].offset
			if rank, ok := c.vocabulary[string(piece[start:end])]; ok {
				return rank
			}
		}
		return math.MaxUint
	}

	for i := 0; i < len(parts)-2; i++ {
		parts[i].rank = getRank(i, 0)
	}

	for {
		if len(parts) == 1 {
			break
		}

		minRank := uint(math.MaxUint)
		minIndex := 0
		for i, p := range parts[:len(parts)-1] {
			if p.rank < minRank {
				minRank = p.rank
				minIndex = i
			}
		}

		if minRank == math.MaxUint {
			break
		}

		parts[minIndex].rank = getRank(minIndex, 1)

		if minIndex > 0 {
			parts[minIndex-1].rank = getRank(minIndex-1, 1)
		}

		parts = append(parts[:minIndex+1], parts[minIndex+2:]...)
	}

	ids := make([]uint, len(parts)-1)
	tokens := make([]string, len(parts)-1)
	for i := 0; i < len(ids); i++ {
		token := string(piece[parts[i].offset:parts[i+1].offset])
		tokens[i] = token
		ids[i] = c.vocabulary[token]
	}
	return ids, tokens
}
