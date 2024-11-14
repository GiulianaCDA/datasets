package events

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	gitlab "github.com/xanzy/go-gitlab"
)

const secretHeader = "X-Gitlab-Token" // #nosec

// DefaultGitlabRequestParserValidator parses and validates GitLab requests.
type DefaultGitlabRequestParserValidator struct{}

// ParseAndValidate returns the JSON payload of the request.
func (d *DefaultGitlabRequestParserValidator) ParseAndValidate(r *http.Request, secret []byte) (interface{}, error) {
	const mergeEventHeader = "Merge Request Hook"
	const noteEventHeader = "Note Hook"

	// Validate secret if specified.
	headerSecret := r.Header.Get(secretHeader)
	secretStr := string(secret)
	if len(secret) != 0 && headerSecret != secretStr {
		return nil, fmt.Errorf("header %s=%s did not match expected secret", secretHeader, headerSecret)
	}

	// Parse request into a gitlab object based on the object type specified
	// in the gitlabHeader.
	bytes, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read request body: %w", err)
	}

	switch r.Header.Get(gitlabHeader) {
	case mergeEventHeader:
		var m gitlab.MergeEvent
		if err := json.Unmarshal(bytes, &m); err != nil {
			return nil, fmt.Errorf("failed to unmarshal merge event: %w", err)
		}
		return m, nil
	case noteEventHeader:
		// First, parse a small part of the json to determine if this is a
		// comment on a merge request or a commit.
		var subset struct {
			ObjectAttributes struct {
				NoteableType string `json:"noteable_type"`
			} `json:"object_attributes"`
		}
		if err := json.Unmarshal(bytes, &subset); err != nil {
			return nil, fmt.Errorf("failed to unmarshal subset: %w", err)
		}

		// We then parse into the correct comment event type.
		switch subset.ObjectAttributes.NoteableType {
		case "Commit":
			var e gitlab.CommitCommentEvent
			if err := json.Unmarshal(bytes, &e); err != nil {
				return nil, fmt.Errorf("failed to unmarshal commit comment event: %w", err)
			}
			return e, nil
		case "MergeRequest":
			var e gitlab.MergeCommentEvent
			if err := json.Unmarshal(bytes, &e); err != nil {
				return nil, fmt.Errorf("failed to unmarshal merge comment event: %w", err)
			}
			return e, nil
		}
	}
	return nil, nil
}
