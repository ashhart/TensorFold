const path = require('node:path')
const { getDefaultConfig } = require('expo/metro-config')

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname)
// Neutral watch protocol is shared with the Mac host outside this independent Expo project.
// Preserve Expo 55's resolver/transformer defaults and watch only that additional source directory.
config.watchFolders = [...new Set([...(config.watchFolders || []), path.resolve(__dirname, '../shared')])]
module.exports = config
