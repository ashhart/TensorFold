package saml

import (
	"errors"
	"fmt"

	"github.com/beevik/etree"
	dsig "github.com/russellhaering/goxmldsig"
)

// Bounds on a message's tree that no IdP comes near; crewjam and goxmldsig walk ancestors, so cost grows with depth.
const (
	maxDepth    = 32
	maxAttrs    = 32
	maxElements = 20000
)

// errMisplaced refuses Signature elements where SAML puts none; nested ones make XML signature checks quadratic.
var errMisplaced = errors.New("signature elements are misplaced")

// errOversized refuses a tree deeper, wider or larger than the bounds, before any library walks it.
var errOversized = errors.New("message nests too deep or holds too many elements or attributes")

// strongAlgorithms are the signature and digest methods Starlight accepts in XML signatures; SHA-1 is refused.
var strongAlgorithms = map[string]map[string]bool{
	"SignatureMethod": {dsig.RSASHA256SignatureMethod: true, dsig.RSASHA384SignatureMethod: true, dsig.RSASHA512SignatureMethod: true,
		dsig.ECDSASHA256SignatureMethod: true, dsig.ECDSASHA384SignatureMethod: true, dsig.ECDSASHA512SignatureMethod: true},
	"DigestMethod": {"http://www.w3.org/2001/04/xmlenc#sha256": true, "http://www.w3.org/2001/04/xmldsig-more#sha384": true,
		"http://www.w3.org/2001/04/xmlenc#sha512": true},
}

// signatures bounds el's tree and lists its Signature elements: one at most under el and under each Assertion child.
func signatures(el *etree.Element) ([]*etree.Element, error) {
	type node struct {
		el    *etree.Element
		depth int
	}
	var out []*etree.Element
	seen := 0
	for todo := []node{{el, 1}}; len(todo) > 0; {
		n := todo[len(todo)-1]
		todo = todo[:len(todo)-1]
		if seen++; seen > maxElements || n.depth > maxDepth || len(n.el.Attr) > maxAttrs {
			return nil, errOversized
		}
		holds := n.el == el || (n.el.Parent() == el && n.el.Tag == "Assertion")
		sigs := 0
		for _, c := range n.el.ChildElements() {
			if c.Tag == "Signature" {
				if sigs++; !holds || sigs > 1 {
					return nil, errMisplaced
				}
				out = append(out, c)
			}
			todo = append(todo, node{c, n.depth + 1})
		}
	}
	return out, nil
}

// weakAlgorithm names a method in sig that is not SHA-256 or stronger, checking every copy a parser might pick.
func weakAlgorithm(sig *etree.Element) string {
	for todo := []*etree.Element{sig}; len(todo) > 0; {
		el := todo[len(todo)-1]
		todo = append(todo[:len(todo)-1], el.ChildElements()...)
		if allowed, ok := strongAlgorithms[el.Tag]; ok {
			for _, a := range el.Attr {
				if a.Key == "Algorithm" && !allowed[a.Value] {
					return a.Value
				}
			}
		}
	}
	return ""
}

// shapeVerifier checks where a response's Signatures sit, and their algorithms, before goxmldsig validates one.
type shapeVerifier struct{}

// VerifySignature refuses misplaced Signature elements under el or a weak one of its own, then validates el's own.
func (shapeVerifier) VerifySignature(vc *dsig.ValidationContext, el *etree.Element) error {
	sigs, err := signatures(el)
	if err != nil {
		return fmt.Errorf("cannot validate signature on %s: %w", el.Tag, err)
	}
	for _, sig := range sigs {
		if alg := weakAlgorithm(sig); alg != "" && sig.Parent() == el {
			return fmt.Errorf("cannot validate signature on %s: algorithm %q is not accepted", el.Tag, alg)
		}
	}
	if _, err := vc.Validate(el); err != nil {
		return fmt.Errorf("cannot validate signature on %s: %w", el.Tag, err)
	}
	return nil
}
