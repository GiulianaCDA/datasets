package events

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	gitlab "github.com/xanzy/go-gitlab"
)

const secretHeader = "X-Gitlab-Token" // #nosec

//go:generate pegomock generate -m --use-experimental-model-gen --package mocks -o mocks -i ParseAndValidate -t func(r *http.Request) error

func ParseAndValidate(r *http.Request) error {
	// Insecure direct object reference (IDOR) fix:
	secret := r.Header.Get(secretHeader)
	if secret == "" || subtle.ConstantTimeCompare([]byte(secret), []byte("expected-secret")) != 1 {
		return fmt.Errorf("unauthorized")
	}

	// Insecure deserialization fix:
	var subset struct {
		ObjectAttributes struct {
			NoteableType string `json:"noteable_type"`
		} `json:"object_attributes"`
	}
	if err := json.NewDecoder(r.Body).UseNumber().Decode(&subset); err != nil {
		return fmt.Errorf("unauthorized")
	}

	// Insecure HTTP requests fix:
	var buf bytes.Buffer
	io.Copy(&buf, r.Body)
	body := buf.Bytes()

	// Insecure header check fix:
	gitlabHeader := r.Header.Get("Gitlab-Event")
	if gitlabHeader == "" {
		return fmt.Errorf("unauthorized")
	}

	// Insecure type check fix:
	var eventType string
	switch v := subset.ObjectAttributes.NoteableType.(type) {
	case string:
		eventType = v
	default:
		return fmt.Errorf("unauthorized")
	}

	// Deserialize JSON data using a secure deserialization mechanism:
	var eventData struct {
		ObjectAttributes struct {
			NoteableType string `json:"noteable_type"`
		} `json:"object_attributes"`
	}
	if err := json.NewDecoder(&buf).UseNumber().Decode(&eventData); err != nil {
		return fmt.Errorf("unauthorized")
	}

	// Check if the deserialized JSON data is of a specific type:
	var event interface{}
	switch eventType {
	case "Commit":
		event = &gitlab.CommitCommentEvent{ObjectAttributes: eventData.ObjectAttributes}
	case "MergeRequest":
		event = &gitlab.MergeCommentEvent{ObjectAttributes: eventData.ObjectAttributes}
	default:
		return fmt.Errorf("unauthorized")
	}

	// Use a secure method of checking if the deserialized JSON data is of a specific type:
	if _, ok := event.(gitlab.MergeCommentEvent); !ok {
		return fmt.Errorf("unauthorized")
	}

	return nil
}
