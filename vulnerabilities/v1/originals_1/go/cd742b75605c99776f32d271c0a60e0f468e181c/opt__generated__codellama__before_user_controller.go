[PYTHON] 
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    tax: float = None

@app.post("/items/")
async def create_item(item: Item):
    return {"name": item.name, "price": item.price, "tax": item.tax}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

if __name__ == "__main__":
    uvicorn.run("tutorial:app", reload=True)
[/PYTHON]
[TYPESCRIPT] 
import { Controller, Post, Body, Get, Param } from '@nestjs/common';
import { UserService } from '../services/user.service';
import { ActionService } from '../services/action.service';
import { EmailService } from '../services/email.service';
import { handler } from '../handler';
import { schema, reason, translator } from '../helper';
import { ErrorHandler } from '../error';
import { errors } from '../errors';
import { ApiKeyAuthGuard } from '../guards/api-key-auth.guard';

@Controller('user')
export class UserController {
  constructor(
    private readonly userService: UserService,
    private readonly actionService: ActionService,
    private readonly emailService: EmailService,
  ) {}

  @Post('change-email')
  async changeEmail(@Body() req: any): Promise<any> {
    const captchaPass = this.actionService.verifyCaptcha(ctx, schema.ActionRecordTypeEmail, ctx.clientIP(), req.captchaId, req.captchaCode);
    if (!captchaPass) {
      throw new ErrorHandler(errors.badRequest, reason.captchaVerificationFailed);
    }
    const resp = await this.userService.changeEmailSendCode(ctx, req);
    return resp;
  }
}
[/TYPESCRIPT]
