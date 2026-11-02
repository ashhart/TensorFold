package fleet

// Normalize applies the defaults and checks the write methods apply, so a caller can compare with what is stored.
func (in ProjectInput) Normalize() (ProjectInput, error) {
	in.defaults()
	return in, in.validate()
}

// Normalize applies the model defaults and checks.
func (in ModelInput) Normalize() (ModelInput, error) {
	err := in.validate()
	return in, err
}

// Normalize applies the deployment defaults and checks, including the engine spec.
func (in DeploymentInput) Normalize() (DeploymentInput, error) {
	err := in.validate()
	return in, err
}

// Normalize applies the provider defaults and checks.
func (in ProviderInput) Normalize() (ProviderInput, error) {
	err := in.validate()
	return in, err
}

// Normalize applies the route defaults for a provider kind and checks.
func (in RouteInput) Normalize(kind string) (RouteInput, error) {
	err := in.validate(kind)
	return in, err
}
