import { UserQuestionService, type UserQuestionServiceOptions } from '../../agent/user-question/service'
export type UserQuestionHost = UserQuestionService
let host: UserQuestionHost | null = null
export function initializeUserQuestionHost(options: UserQuestionServiceOptions): UserQuestionHost { return host ??= new UserQuestionService(options) }
export function getUserQuestionHost(): UserQuestionHost | null { return host }
