package web

// Link is a button on a message page.
type Link struct {
	Href  string
	Label string
}

// MessagePage is a titled message; Kind is info, success or error.
type MessagePage struct {
	Kind    string
	Heading string
	Body    string
	Links   []Link
}

// LoginOption is one sign-in button.
type LoginOption struct {
	Slug     string
	Name     string
	Preset   string
	Protocol string
}

// LoginPage lists connections and offers email routing; SetupCode switches it to owner claiming.
type LoginPage struct {
	Options   []LoginOption
	ShowEmail bool
	Email     string
	ReturnTo  string
	Error     string
	Notice    string
	SetupCode string
}

// Hidden is a form field carried through an approval page.
type Hidden struct {
	Name  string
	Value string
}

// ApprovePage asks a signed-in user to approve an OAuth client.
type ApprovePage struct {
	ClientName string
	DeviceName string
	UserEmail  string
	Scopes     []string
	Action     string
	CSRF       string
	Hidden     []Hidden
}

// DevicePage collects or confirms a device-flow user code.
type DevicePage struct {
	Action     string
	CSRF       string
	UserCode   string
	Error      string
	Confirm    bool
	ClientName string
	DeviceName string
}

// RecoverPage confirms a break-glass sign-in.
type RecoverPage struct {
	Code  string
	Error string
}

// PresetChoice is one identity provider offered by the setup wizard.
type PresetChoice struct {
	Name      string
	Label     string
	Protocols []string
}

// FormField is one input in the setup wizard.
type FormField struct {
	Name        string
	Label       string
	Help        string
	Kind        string
	Required    bool
	Placeholder string
	Options     []string
	Value       string
}

// SetupConnection shows the URLs an administrator registers at the identity provider.
type SetupConnection struct {
	Slug        string
	Name        string
	Protocol    string
	RedirectURI string
	ACSURL      string
	EntityID    string
	MetadataURL string
}

// SetupPage drives first-owner setup; Step is choose, configure or signin.
type SetupPage struct {
	Code        string
	Step        string
	Error       string
	Notice      string
	PublicURL   string
	Presets     []PresetChoice
	Preset      string
	PresetLabel string
	Protocol    string
	Fields      []FormField
	Connections []SetupConnection
}
